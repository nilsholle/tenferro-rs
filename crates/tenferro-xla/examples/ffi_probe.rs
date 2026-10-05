//! Calls a foreign kernel from a StableHLO program through the PJRT FFI
//! extension: the uniform-1d segmented polynomial of NVIDIA cuEquivariance
//! (`libcue_ops_jax.so`, an optional proprietary library that the user
//! installs; its kernel interface is described by the headers it ships,
//! `cuequivariance_ops/equivariance/uniform1d/api.hh`).
//!
//! ```text
//! cargo run --release --features pjrt --example ffi_probe -- \
//!     libpjrt_cuda.so libcue_ops_jax.so
//! ```
//!
//! The polynomial `z[b,k,u] = sum_paths c x[b,i,u] y[b,j]` is evaluated by
//! the kernel and compared with the same sum on the host.

use std::ffi::{c_char, c_void, CString};

use tenferro_xla::PjrtSession;

type Compile = unsafe extern "C" fn(
    *const c_char, // name
    i64,           // math dtype
    i64,           // operand extent
    i64,           // inputs
    i64,           // outputs
    i64,           // index buffers
    *const i64,
    i64, // batch sizes
    *const i64,
    i64, // batch dimension kind per buffer and batch axis
    *const i64,
    i64, // segments per buffer
    *const i64,
    i64, // segment kind per buffer (scalar or vector)
    *const i64,
    i64, // index configuration
    *const i64,
    i64, // index extents
    *const i64,
    i64, // dtype per buffer
    *const i64,
    i64, // operands per operation
    *const i64,
    i64, // buffers of the operations
    *const i64,
    i64, // paths per operation
    *const i64,
    i64, // first path index entry per operation
    *const i64,
    i64, // first coefficient per operation
    *const i64,
    i64, // segment indices of the paths
    *const i64,
    i64, // coefficients (float64 bit patterns)
) -> i64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let plugin = args.next().ok_or("plugin path")?;
    let library = args.next().ok_or("path of libcue_ops_jax.so")?;
    let session = PjrtSession::open(plugin)?;

    // SAFETY: loading the library runs its initialisers; it has to stay
    // loaded as long as the plugin, so it is leaked.
    let library = Box::leak(Box::new(unsafe { libloading::Library::new(&library)? }));
    let compile: libloading::Symbol<Compile> =
        unsafe { library.get(b"compile_uniform_1d_ctypes")? };
    // (a `Symbol<fn>` dereferences to the function, i.e. to its address)
    let handler: libloading::Symbol<unsafe extern "C" fn()> =
        unsafe { library.get(b"execute_uniform_1d_cuda_handler")? };
    let handler = *handler as *mut c_void;

    // x: 2 segments of extent 4, y: 3 scalar segments, z: 2 segments
    let (batch, extent) = (5usize, 4usize);
    let paths: [([i64; 3], f64); 3] = [([0, 0, 0], 1.0), ([1, 2, 1], 0.5), ([0, 1, 1], 2.0)];
    let path_indices: Vec<i64> = paths.iter().flat_map(|(i, _)| *i).collect();
    let coefficients: Vec<i64> = paths.iter().map(|(_, c)| c.to_bits() as i64).collect();
    let list = |values: &[i64]| (values.as_ptr(), values.len() as i64);
    let name = CString::new("ffi_probe")?;
    let (b0, b1) = list(&[batch as i64]);
    let (d0, d1) = list(&[0, 0, 0]); // batched
    let (s0, s1) = list(&[2, 3, 2]);
    let (k0, k1) = list(&[1, 0, 1]); // vector, scalar, vector
    let (c0, c1) = list(&[-1, -1, -1]); // no index buffers
    let (e0, e1) = list(&[]);
    let (t0, t1) = list(&[0, 0, 0]); // float32
    let (o0, o1) = list(&[3]);
    let (ob0, ob1) = list(&[0, 1, 2]);
    let (p0, p1) = list(&[paths.len() as i64]);
    let (si0, si1) = list(&[0]);
    let (sc0, sc1) = list(&[0]);
    let (pi0, pi1) = list(&path_indices);
    let (pc0, pc1) = list(&coefficients);
    // SAFETY: the argument list follows the C entry point of the library.
    let handle = unsafe {
        compile(
            name.as_ptr(),
            0,
            extent as i64,
            2,
            1,
            0,
            b0,
            b1,
            d0,
            d1,
            s0,
            s1,
            k0,
            k1,
            c0,
            c1,
            e0,
            e1,
            t0,
            t1,
            o0,
            o1,
            ob0,
            ob1,
            p0,
            p1,
            si0,
            si1,
            sc0,
            sc1,
            pi0,
            pi1,
            pc0,
            pc1,
        )
    };
    println!("kernel handle: {handle}");

    // SAFETY: the symbol is an XLA FFI handler of the leaked library.
    unsafe { session.register_ffi_handler("uniform_1d_cuda", handler, "CUDA", false)? };

    let mlir = format!(
        "module {{\n  func.func @main(%arg0: tensor<{batch}x2x{extent}xf32>, %arg1: tensor<{batch}x3x1xf32>) -> tensor<{batch}x2x{extent}xf32> {{\n    %0 = stablehlo.custom_call @uniform_1d_cuda(%arg0, %arg1) {{api_version = 4 : i32, backend_config = {{first_batch_size = {batch} : i64, handle = {handle} : i64, num_indices = 0 : i64}}}} : (tensor<{batch}x2x{extent}xf32>, tensor<{batch}x3x1xf32>) -> tensor<{batch}x2x{extent}xf32>\n    return %0 : tensor<{batch}x2x{extent}xf32>\n  }}\n}}\n"
    );
    let program = session.compile(&mlir)?;
    let x: Vec<f32> = (0..batch * 2 * extent)
        .map(|i| (i as f32 * 0.37).sin())
        .collect();
    let y: Vec<f32> = (0..batch * 3).map(|i| (i as f32 * 0.91).cos()).collect();
    let out = program.execute(&[
        &session.upload(&[batch, 2, extent], &x)?,
        &session.upload(&[batch, 3, 1], &y)?,
    ])?;
    let z: Vec<f32> = out[0].to_vec()?;

    let mut expected = vec![0.0f32; batch * 2 * extent];
    for b in 0..batch {
        for ([i, j, k], c) in paths {
            for u in 0..extent {
                expected[(b * 2 + k as usize) * extent + u] +=
                    c as f32 * x[(b * 2 + i as usize) * extent + u] * y[b * 3 + j as usize];
            }
        }
    }
    println!("kernel:   {:?}", &z[..8]);
    println!("expected: {:?}", &expected[..8]);
    let worst = z
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("largest deviation from the host sum: {worst:.2e}");
    if worst > 1e-5 {
        return Err("the kernel result differs".into());
    }
    println!("FFI_PROBE_OK");
    Ok(())
}
