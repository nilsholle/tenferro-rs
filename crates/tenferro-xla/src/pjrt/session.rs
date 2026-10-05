//! Persistent PJRT session (fastatomstruct patch on top of tenferro-xla 0.7.1).
//!
//! `run_many_with_inputs` creates a client and recompiles on every call, which
//! is fine for a smoke test and unusable for a hot loop. This module keeps the
//! client alive, compiles a module once into a [`PjrtProgram`], and lets
//! callers keep operands on the device as [`DeviceBuffer`]s between calls.
//!
//! Host data crosses this boundary as dense **row-major** slices: that is the
//! only host layout every PJRT client (CUDA included) accepts.

use std::ffi::{c_char, c_void};
use std::mem;
use std::ptr;
use std::slice;
use std::sync::Arc;

use crate::{Error, Result};

use super::plugin::PjrtPlugin;
use super::sys::*;

#[repr(C)]
struct PluginInitializeArgs {
    struct_size: usize,
    extension_start: *mut PJRT_Extension_Base,
}

type PluginInitializeFn = unsafe extern "C" fn(*mut PluginInitializeArgs) -> *mut PJRT_Error;

/// Element types that can cross the host/device boundary.
pub trait HostScalar: Copy + Default + 'static {
    #[doc(hidden)]
    const PJRT_TYPE: u32;
}

impl HostScalar for f32 {
    const PJRT_TYPE: u32 = PJRT_Buffer_Type::F32 as u32;
}
impl HostScalar for f64 {
    const PJRT_TYPE: u32 = PJRT_Buffer_Type::F64 as u32;
}
impl HostScalar for i32 {
    const PJRT_TYPE: u32 = PJRT_Buffer_Type::S32 as u32;
}
impl HostScalar for i64 {
    const PJRT_TYPE: u32 = PJRT_Buffer_Type::S64 as u32;
}

fn buffer_type(code: u32) -> PJRT_Buffer_Type {
    match code {
        4 => PJRT_Buffer_Type::S32,
        5 => PJRT_Buffer_Type::S64,
        11 => PJRT_Buffer_Type::F32,
        _ => PJRT_Buffer_Type::F64,
    }
}

struct Inner {
    client: *mut PJRT_Client,
    device: *mut PJRT_Device,
    // Declared last: the client must be destroyed before the library unloads.
    plugin: PjrtPlugin,
}

// SAFETY: PJRT clients, executables and buffers are documented as
// thread-compatible handles guarded by the plugin; we only ever hand out
// shared references and the plugin outlives every handle via `Arc<Inner>`.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Inner {
    fn api(&self) -> &PJRT_Api {
        // SAFETY: the table is owned by the loaded plugin, which `self` keeps alive.
        unsafe { &*self.plugin.api() }
    }

    fn error_message(&self, error: *mut PJRT_Error) -> String {
        let api = self.api();
        let mut args = PJRT_Error_Message_Args {
            struct_size: mem::size_of::<PJRT_Error_Message_Args>(),
            extension_start: ptr::null_mut(),
            error,
            message: ptr::null(),
            message_size: 0,
        };
        unsafe { (api.pjrt_error_message)(&mut args) };
        let message = if args.message.is_null() {
            "unknown PJRT error".to_string()
        } else {
            let bytes =
                unsafe { slice::from_raw_parts(args.message.cast::<u8>(), args.message_size) };
            String::from_utf8_lossy(bytes).into_owned()
        };
        let mut destroy = PJRT_Error_Destroy_Args {
            struct_size: mem::size_of::<PJRT_Error_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            error,
        };
        unsafe { (api.pjrt_error_destroy)(&mut destroy) };
        message
    }

    fn check(&self, call: &'static str, error: *mut PJRT_Error) -> Result<()> {
        if error.is_null() {
            Ok(())
        } else {
            Err(Error::PjrtCall {
                call,
                message: self.error_message(error),
            })
        }
    }

    fn await_event(&self, call: &'static str, event: *mut PJRT_Event) -> Result<()> {
        if event.is_null() {
            return Ok(());
        }
        let api = self.api();
        let mut args = PJRT_Event_Await_Args {
            struct_size: mem::size_of::<PJRT_Event_Await_Args>(),
            extension_start: ptr::null_mut(),
            event,
        };
        let error = unsafe { (api.pjrt_event_await)(&mut args) };
        let mut destroy = PJRT_Event_Destroy_Args {
            struct_size: mem::size_of::<PJRT_Event_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            event,
        };
        let _ = unsafe { (api.pjrt_event_destroy)(&mut destroy) };
        self.check(call, error)
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        if self.client.is_null() {
            return;
        }
        let mut args = PJRT_Client_Destroy_Args {
            struct_size: mem::size_of::<PJRT_Client_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            client: self.client,
        };
        let _ = unsafe { (self.api().pjrt_client_destroy)(&mut args) };
    }
}

/// A PJRT client bound to its first addressable device.
#[derive(Clone)]
pub struct PjrtSession {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for PjrtSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PjrtSession")
            .field("plugin", &self.inner.plugin.path())
            .finish_non_exhaustive()
    }
}

impl PjrtSession {
    /// Initialise the plugin and create a client on its first device.
    ///
    /// # Errors
    ///
    /// Returns `Error::PjrtCall` when the plugin rejects initialisation,
    /// client creation, or reports no addressable device.
    pub fn new(plugin: PjrtPlugin) -> Result<Self> {
        let api_ptr = plugin.api();
        if api_ptr.is_null() {
            return Err(Error::PjrtCall {
                call: "GetPjrtApi",
                message: "plugin returned null API table".to_string(),
            });
        }
        let mut inner = Inner {
            client: ptr::null_mut(),
            device: ptr::null_mut(),
            plugin,
        };
        let init = inner.api().pjrt_plugin_initialize;
        if !init.is_null() {
            // SAFETY: slot 6 of the PJRT API table is `PJRT_Plugin_Initialize`.
            let init: PluginInitializeFn = unsafe { mem::transmute(init) };
            let mut args = PluginInitializeArgs {
                struct_size: mem::size_of::<PluginInitializeArgs>(),
                extension_start: ptr::null_mut(),
            };
            let error = unsafe { init(&mut args) };
            inner.check("PJRT_Plugin_Initialize", error)?;
        }

        let mut args = PJRT_Client_Create_Args {
            struct_size: mem::size_of::<PJRT_Client_Create_Args>(),
            extension_start: ptr::null_mut(),
            create_options: ptr::null(),
            num_options: 0,
            kv_get_callback: ptr::null(),
            kv_get_user_arg: ptr::null_mut(),
            kv_put_callback: ptr::null(),
            kv_put_user_arg: ptr::null_mut(),
            client: ptr::null_mut(),
            kv_try_get_callback: ptr::null(),
            kv_try_get_user_arg: ptr::null_mut(),
        };
        let error = unsafe { (inner.api().pjrt_client_create)(&mut args) };
        inner.check("PJRT_Client_Create", error)?;
        if args.client.is_null() {
            return Err(Error::PjrtCall {
                call: "PJRT_Client_Create",
                message: "returned null client".to_string(),
            });
        }
        inner.client = args.client;

        let mut devices = PJRT_Client_AddressableDevices_Args {
            struct_size: mem::size_of::<PJRT_Client_AddressableDevices_Args>(),
            extension_start: ptr::null_mut(),
            client: inner.client,
            addressable_devices: ptr::null(),
            num_addressable_devices: 0,
        };
        let error = unsafe { (inner.api().pjrt_client_addressable_devices)(&mut devices) };
        inner.check("PJRT_Client_AddressableDevices", error)?;
        if devices.num_addressable_devices == 0 || devices.addressable_devices.is_null() {
            return Err(Error::PjrtCall {
                call: "PJRT_Client_AddressableDevices",
                message: "client has no addressable devices".to_string(),
            });
        }
        inner.device = unsafe { *devices.addressable_devices };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Load a plugin from `path` and open a session on it.
    ///
    /// # Errors
    ///
    /// Propagates plugin-load and [`PjrtSession::new`] errors.
    pub fn open(path: impl Into<std::path::PathBuf>) -> Result<Self> {
        Self::new(PjrtPlugin::load_path(path)?)
    }

    /// Register a foreign function (XLA FFI handler) under `name`, so that
    /// programs compiled afterwards can call it with
    /// `stablehlo.custom_call @name(...) {api_version = 4 : i32, ...}`.
    /// `platform`: the XLA platform the handler is for (`"CUDA"`, `"Host"`).
    /// `command_buffer_compatible`: the handler may run inside a command
    /// buffer (CUDA graph).
    ///
    /// # Safety
    ///
    /// `handler` has to be an `XLA_FFI_Handler` (`XLA_FFI_Error*
    /// (*)(XLA_FFI_CallFrame*)`) of a library that stays loaded as long as
    /// the plugin, built for the FFI ABI of this plugin.
    ///
    /// # Errors
    ///
    /// `Error::PjrtCall` if the plugin has no FFI extension or rejects the
    /// registration.
    pub unsafe fn register_ffi_handler(
        &self,
        name: &str,
        handler: *mut std::ffi::c_void,
        platform: &str,
        command_buffer_compatible: bool,
    ) -> Result<()> {
        let api = self.inner.api();
        let mut extension = api.extension_start;
        // SAFETY: the extension list is owned by the plugin; every entry
        // starts with a `PJRT_Extension_Base`.
        while !extension.is_null()
            && unsafe { (*extension).extension_type } != PJRT_EXTENSION_TYPE_FFI
        {
            extension = unsafe { (*extension).next };
        }
        let missing = || Error::PjrtCall {
            call: "PJRT_FFI_Register_Handler",
            message: "the plugin has no FFI extension".to_string(),
        };
        if extension.is_null() {
            return Err(missing());
        }
        // SAFETY: an extension of type FFI is a `PJRT_FFI_Extension`.
        let register = unsafe { (*extension.cast::<PJRT_FFI_Extension>()).register_handler }
            .ok_or_else(missing)?;
        let mut args = PJRT_FFI_Register_Handler_Args {
            struct_size: mem::size_of::<PJRT_FFI_Register_Handler_Args>(),
            target_name: name.as_ptr().cast(),
            target_name_size: name.len(),
            handler,
            platform_name: platform.as_ptr().cast(),
            platform_name_size: platform.len(),
            traits: u32::from(command_buffer_compatible),
        };
        // SAFETY: `args` matches the plugin's argument struct; the strings
        // are passed with their lengths.
        let error = unsafe { register(&mut args) };
        self.inner.check("PJRT_FFI_Register_Handler", error)
    }

    /// Peak number of bytes the device allocator has had in use since the
    /// client was created, if the plugin reports it (the CPU plugin does not).
    pub fn peak_device_bytes(&self) -> Option<u64> {
        let inner = &self.inner;
        // SAFETY: an all-zero bit pattern is valid for this plain C struct.
        let mut args: PJRT_Device_MemoryStats_Args = unsafe { std::mem::zeroed() };
        args.struct_size = std::mem::size_of::<PJRT_Device_MemoryStats_Args>();
        args.device = inner.device;
        // SAFETY: `args` is a correctly sized argument block and the device
        // handle is owned by the live client.
        let error = unsafe { (inner.api().pjrt_device_memory_stats)(&mut args) };
        if inner.check("PJRT_Device_MemoryStats", error).is_err() {
            return None;
        }
        let bytes = if args.peak_bytes_in_use_is_set {
            args.peak_bytes_in_use
        } else {
            args.bytes_in_use
        };
        u64::try_from(bytes).ok()
    }

    /// Compile StableHLO MLIR text into a reusable program.
    ///
    /// # Errors
    ///
    /// Returns `Error::PjrtCall` with the compiler diagnostic on failure.
    pub fn compile(&self, mlir: &str) -> Result<PjrtProgram> {
        let inner = &self.inner;
        let mut code = mlir.as_bytes().to_vec();
        let format = b"mlir";
        let mut program = PJRT_Program {
            struct_size: mem::size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: code.as_mut_ptr().cast::<c_char>(),
            code_size: code.len(),
            format: format.as_ptr().cast::<c_char>(),
            format_size: format.len(),
        };
        // xla.CompileOptionsProto { executable_build_options(3) {
        //   num_replicas(4) = 1, num_partitions(5) = 1 } }
        let compile_options: [u8; 6] = [0x1a, 0x04, 0x20, 0x01, 0x28, 0x01];
        let mut args = PJRT_Client_Compile_Args {
            struct_size: mem::size_of::<PJRT_Client_Compile_Args>(),
            extension_start: ptr::null_mut(),
            client: inner.client,
            program: &mut program,
            compile_options: compile_options.as_ptr().cast::<c_char>(),
            compile_options_size: compile_options.len(),
            executable: ptr::null_mut(),
        };
        let error = unsafe { (inner.api().pjrt_client_compile)(&mut args) };
        inner.check("PJRT_Client_Compile", error)?;
        if args.executable.is_null() {
            return Err(Error::PjrtCall {
                call: "PJRT_Client_Compile",
                message: "returned null executable".to_string(),
            });
        }
        let mut program = PjrtProgram {
            inner: Arc::clone(inner),
            ptr: args.executable,
            num_outputs: 0,
        };

        let mut get = PJRT_LoadedExecutable_GetExecutable_Args {
            struct_size: mem::size_of::<PJRT_LoadedExecutable_GetExecutable_Args>(),
            extension_start: ptr::null_mut(),
            loaded_executable: program.ptr,
            executable: ptr::null_mut(),
        };
        let error = unsafe { (inner.api().pjrt_loaded_executable_get_executable)(&mut get) };
        inner.check("PJRT_LoadedExecutable_GetExecutable", error)?;
        let mut count = PJRT_Executable_NumOutputs_Args {
            struct_size: mem::size_of::<PJRT_Executable_NumOutputs_Args>(),
            extension_start: ptr::null_mut(),
            executable: get.executable,
            num_outputs: 0,
        };
        let error = unsafe { (inner.api().pjrt_executable_num_outputs)(&mut count) };
        let mut destroy = PJRT_Executable_Destroy_Args {
            struct_size: mem::size_of::<PJRT_Executable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            executable: get.executable,
        };
        let _ = unsafe { (inner.api().pjrt_executable_destroy)(&mut destroy) };
        inner.check("PJRT_Executable_NumOutputs", error)?;
        program.num_outputs = count.num_outputs;
        Ok(program)
    }

    /// Copy a dense row-major host slice to the device.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidProgram` when `data.len()` does not match
    /// `shape`, or `Error::PjrtCall` for a plugin failure.
    pub fn upload<T: HostScalar>(&self, shape: &[usize], data: &[T]) -> Result<DeviceBuffer> {
        let count: usize = shape.iter().product();
        if count != data.len() {
            return Err(Error::InvalidProgram {
                message: format!(
                    "upload: shape {shape:?} needs {count} elements, got {}",
                    data.len()
                ),
            });
        }
        let inner = &self.inner;
        let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let mut args = PJRT_Client_BufferFromHostBuffer_Args {
            struct_size: mem::size_of::<PJRT_Client_BufferFromHostBuffer_Args>(),
            extension_start: ptr::null_mut(),
            client: inner.client,
            data: data.as_ptr().cast::<c_void>(),
            type_: buffer_type(T::PJRT_TYPE),
            dims: dims.as_ptr(),
            num_dims: dims.len(),
            byte_strides: ptr::null(),
            num_byte_strides: 0,
            host_buffer_semantics: PJRT_HostBufferSemantics::ImmutableOnlyDuringCall,
            device: inner.device,
            memory: ptr::null_mut(),
            device_layout: ptr::null_mut(),
            done_with_host_buffer: ptr::null_mut(),
            buffer: ptr::null_mut(),
        };
        let error = unsafe { (inner.api().pjrt_client_buffer_from_host_buffer)(&mut args) };
        inner.check("PJRT_Client_BufferFromHostBuffer", error)?;
        inner.await_event(
            "PJRT_Client_BufferFromHostBuffer.done_with_host_buffer",
            args.done_with_host_buffer,
        )?;
        if args.buffer.is_null() {
            return Err(Error::PjrtCall {
                call: "PJRT_Client_BufferFromHostBuffer",
                message: "returned null buffer".to_string(),
            });
        }
        Ok(DeviceBuffer {
            inner: Arc::clone(inner),
            ptr: args.buffer,
        })
    }
}

/// A compiled, loaded executable.
pub struct PjrtProgram {
    inner: Arc<Inner>,
    ptr: *mut PJRT_LoadedExecutable,
    num_outputs: usize,
}

// SAFETY: see `Inner`; execution takes `&self` and PJRT serialises internally.
unsafe impl Send for PjrtProgram {}
unsafe impl Sync for PjrtProgram {}

impl PjrtProgram {
    /// Number of results `execute` returns.
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Run the program on device buffers; results stay on the device.
    ///
    /// # Errors
    ///
    /// Returns `Error::PjrtCall` for arity/shape mismatches reported by the
    /// plugin or a failed execution.
    pub fn execute(&self, inputs: &[&DeviceBuffer]) -> Result<Vec<DeviceBuffer>> {
        let inner = &self.inner;
        let input_ptrs: Vec<*mut PJRT_Buffer> = inputs.iter().map(|b| b.ptr).collect();
        let argument_lists = [input_ptrs.as_ptr()];
        let mut output_ptrs = vec![ptr::null_mut(); self.num_outputs];
        let output_lists = [output_ptrs.as_mut_ptr()];
        let mut complete_events = [ptr::null_mut()];
        // Inputs are reused across calls (weights, topology): never donate.
        let non_donatable: Vec<i64> = (0..inputs.len() as i64).collect();
        let mut options = PJRT_ExecuteOptions {
            struct_size: mem::size_of::<PJRT_ExecuteOptions>(),
            extension_start: ptr::null_mut(),
            send_callbacks: ptr::null_mut(),
            recv_callbacks: ptr::null_mut(),
            num_send_ops: 0,
            num_recv_ops: 0,
            launch_id: 0,
            non_donatable_input_indices: non_donatable.as_ptr(),
            num_non_donatable_input_indices: non_donatable.len(),
            context: ptr::null_mut(),
            call_location: ptr::null(),
            num_tasks: 0,
            task_ids: ptr::null_mut(),
            incarnation_ids: ptr::null_mut(),
            multi_slice_config: ptr::null_mut(),
        };
        let mut args = PJRT_LoadedExecutable_Execute_Args {
            struct_size: mem::size_of::<PJRT_LoadedExecutable_Execute_Args>(),
            extension_start: ptr::null_mut(),
            executable: self.ptr,
            options: &mut options,
            argument_lists: argument_lists.as_ptr(),
            num_devices: 1,
            num_args: input_ptrs.len(),
            output_lists: output_lists.as_ptr(),
            device_complete_events: complete_events.as_mut_ptr(),
            execute_device: inner.device,
        };
        let error = unsafe { (inner.api().pjrt_loaded_executable_execute)(&mut args) };
        inner.check("PJRT_LoadedExecutable_Execute", error)?;
        inner.await_event(
            "PJRT_LoadedExecutable_Execute.device_complete",
            complete_events[0],
        )?;
        output_ptrs
            .into_iter()
            .enumerate()
            .map(|(index, ptr)| {
                if ptr.is_null() {
                    Err(Error::PjrtCall {
                        call: "PJRT_LoadedExecutable_Execute",
                        message: format!("output buffer {index} is null"),
                    })
                } else {
                    Ok(DeviceBuffer {
                        inner: Arc::clone(inner),
                        ptr,
                    })
                }
            })
            .collect()
    }
}

impl Drop for PjrtProgram {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let mut args = PJRT_LoadedExecutable_Destroy_Args {
            struct_size: mem::size_of::<PJRT_LoadedExecutable_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            executable: self.ptr,
        };
        let _ = unsafe { (self.inner.api().pjrt_loaded_executable_destroy)(&mut args) };
    }
}

/// A tensor living on the session's device.
pub struct DeviceBuffer {
    inner: Arc<Inner>,
    ptr: *mut PJRT_Buffer,
}

// SAFETY: see `Inner`.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    /// Copy the buffer back as a dense row-major vector.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidProgram` when the buffer's byte size is not a
    /// multiple of `size_of::<T>()`, or `Error::PjrtCall` for a plugin failure.
    pub fn to_vec<T: HostScalar>(&self) -> Result<Vec<T>> {
        let inner = &self.inner;
        let api = inner.api();
        // First call with a null destination only reports the required size.
        let mut query = PJRT_Buffer_ToHostBuffer_Args {
            struct_size: mem::size_of::<PJRT_Buffer_ToHostBuffer_Args>(),
            extension_start: ptr::null_mut(),
            src: self.ptr,
            host_layout: ptr::null_mut(),
            dst: ptr::null_mut(),
            dst_size: 0,
            event: ptr::null_mut(),
        };
        let error = unsafe { (api.pjrt_buffer_to_host_buffer)(&mut query) };
        inner.check("PJRT_Buffer_ToHostBuffer(size)", error)?;
        inner.await_event("PJRT_Buffer_ToHostBuffer(size).event", query.event)?;
        let bytes = query.dst_size;
        if !bytes.is_multiple_of(mem::size_of::<T>()) {
            return Err(Error::InvalidProgram {
                message: format!(
                    "device buffer of {bytes} bytes is not a whole number of {}-byte elements",
                    mem::size_of::<T>()
                ),
            });
        }
        let mut output = vec![T::default(); bytes / mem::size_of::<T>()];
        let mut args = PJRT_Buffer_ToHostBuffer_Args {
            struct_size: mem::size_of::<PJRT_Buffer_ToHostBuffer_Args>(),
            extension_start: ptr::null_mut(),
            src: self.ptr,
            host_layout: ptr::null_mut(),
            dst: output.as_mut_ptr().cast::<c_void>(),
            dst_size: bytes,
            event: ptr::null_mut(),
        };
        let error = unsafe { (api.pjrt_buffer_to_host_buffer)(&mut args) };
        inner.check("PJRT_Buffer_ToHostBuffer", error)?;
        inner.await_event("PJRT_Buffer_ToHostBuffer.event", args.event)?;
        Ok(output)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let mut args = PJRT_Buffer_Destroy_Args {
            struct_size: mem::size_of::<PJRT_Buffer_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            buffer: self.ptr,
        };
        let _ = unsafe { (self.inner.api().pjrt_buffer_destroy)(&mut args) };
    }
}
