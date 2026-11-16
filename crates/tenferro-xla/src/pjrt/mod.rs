use std::path::PathBuf;

use crate::{Error, Result};

mod execute;
mod plugin;
mod session;
mod sys;

pub(crate) use execute::run_many_with_inputs;
pub use plugin::PjrtPlugin;
pub use session::{DeviceBuffer, HostScalar, PjrtProgram, PjrtSession};

pub(crate) fn plugin_path_from_env(var: &'static str) -> Result<PathBuf> {
    std::env::var_os(var)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or(Error::MissingEnv { var })
}

/// Client create options of the GPU plugin from the environment, so that a
/// large system can use more than XLA's default 75 % of the device memory:
/// `TENFERRO_XLA_MEMORY_FRACTION` (float, option `memory_fraction`),
/// `TENFERRO_XLA_ALLOCATOR` (`bfc`, `cuda_async`, `platform`, `default`) and
/// `TENFERRO_XLA_PREALLOCATE` (`0`/`1`). Unset variables pass no option;
/// plugins that do not know an option reject it, so set them for GPU only.
pub(crate) struct ClientOptions {
    names: Vec<&'static str>,
    strings: Vec<String>,
    values: Vec<sys::PJRT_NamedValue>,
}

impl ClientOptions {
    pub(crate) fn from_env() -> Result<Self> {
        let get = |var: &str| std::env::var(var).ok().filter(|v| !v.trim().is_empty());
        let mut options = Self {
            names: Vec::new(),
            strings: Vec::new(),
            values: Vec::new(),
        };
        // (name, kind, raw value, string index)
        let mut raw: Vec<(&'static str, i32, u64, Option<usize>)> = Vec::new();
        if let Some(v) = get("TENFERRO_XLA_MEMORY_FRACTION") {
            let f: f32 = v.trim().parse().map_err(|_| Error::PjrtCall {
                call: "PJRT_Client_Create",
                message: format!("TENFERRO_XLA_MEMORY_FRACTION={v} is not a number"),
            })?;
            raw.push(("memory_fraction", 3, f.to_bits() as u64, None));
        }
        if let Some(v) = get("TENFERRO_XLA_PREALLOCATE") {
            let b = matches!(v.trim(), "1" | "true" | "True" | "yes");
            raw.push(("preallocate", 4, b as u64, None));
        }
        if let Some(v) = get("TENFERRO_XLA_ALLOCATOR") {
            options.strings.push(v.trim().to_string());
            raw.push(("allocator", 0, 0, Some(options.strings.len() - 1)));
        }
        for (name, kind, value, string) in raw {
            options.names.push(name);
            let (value, value_size) = match string {
                Some(i) => (options.strings[i].as_ptr() as u64, options.strings[i].len()),
                None => (value, 1),
            };
            options.values.push(sys::PJRT_NamedValue {
                struct_size: std::mem::size_of::<sys::PJRT_NamedValue>(),
                extension_start: std::ptr::null_mut(),
                name: name.as_ptr().cast(),
                name_size: name.len(),
                kind,
                value,
                value_size,
            });
        }
        Ok(options)
    }

    pub(crate) fn as_ptr(&self) -> *const std::ffi::c_void {
        if self.values.is_empty() {
            std::ptr::null()
        } else {
            self.values.as_ptr().cast()
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }
}
