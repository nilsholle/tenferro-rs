//! Lowering for the indexing / structural / comparison part of the core op
//! vocabulary (fastatomstruct patch on top of tenferro-xla 0.7.1).
//!
//! Every op here is emitted in MLIR *generic* form so that attribute spelling
//! does not depend on the StableHLO pretty-printer version of the plugin.

use tenferro_tensor::{CompareDir, DType, GatherConfig, PadConfig, ScatterConfig, SliceConfig};

use crate::{Error, Result};

use super::super::emit::Emitter;
use super::super::types::{format_tensor_type, TensorType};
use super::{require_input_count, Value};

fn i64_list<T: ToString>(values: &[T]) -> String {
    let inner = values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    if inner.is_empty() {
        "array<i64>".to_string()
    } else {
        format!("array<i64: {inner}>")
    }
}

fn bracket_list<T: ToString>(values: &[T]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn out(name: String, ty: &TensorType) -> Value {
    Value {
        name,
        ty: ty.clone(),
    }
}

/// Zero literal for a dense scalar constant of `dtype`.
pub(super) fn scalar_literal(dtype: DType, value: f64) -> String {
    match dtype {
        DType::Bool => if value != 0.0 { "true" } else { "false" }.to_string(),
        DType::I32 | DType::I64 => format!("{}", value as i64),
        _ => super::format_float(value),
    }
}

pub(super) fn emit_scalar_constant(
    dtype: DType,
    value: f64,
    emitter: &mut Emitter,
) -> Result<Value> {
    let ty = TensorType::scalar(dtype, "scalar constant")?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = stablehlo.constant dense<{}> : {}",
        scalar_literal(dtype, value),
        format_tensor_type(&ty)
    ));
    Ok(Value { name, ty })
}

/// Bring `input` to `target` by dtype conversion and NumPy (right-aligned)
/// broadcasting. tenferro binary ops promote dtypes and broadcast implicitly;
/// StableHLO does neither.
pub(super) fn coerce(input: &Value, target: &TensorType, emitter: &mut Emitter) -> Result<Value> {
    let mut value = input.clone();
    if value.ty.dtype != target.dtype {
        let ty = TensorType::new(value.ty.shape.clone(), target.dtype, "implicit convert")?;
        let name = emitter.value();
        emitter.line(format!(
            "{name} = stablehlo.convert {} : ({}) -> {}",
            value.name,
            format_tensor_type(&value.ty),
            format_tensor_type(&ty)
        ));
        value = Value { name, ty };
    }
    if value.ty.shape != target.shape {
        let in_rank = value.ty.shape.len();
        let out_rank = target.shape.len();
        if in_rank > out_rank {
            return Err(Error::InvalidProgram {
                message: format!(
                    "cannot broadcast {:?} to lower-rank {:?}",
                    value.ty.shape, target.shape
                ),
            });
        }
        let diff = out_rank - in_rank;
        for (axis, &dim) in value.ty.shape.iter().enumerate() {
            let want = target.shape[axis + diff];
            if dim != want && dim != 1 {
                return Err(Error::InvalidProgram {
                    message: format!(
                        "cannot broadcast {:?} to {:?}",
                        value.ty.shape, target.shape
                    ),
                });
            }
        }
        let dims: Vec<usize> = (diff..out_rank).collect();
        let name = emitter.value();
        emitter.line(format!(
            "{name} = stablehlo.broadcast_in_dim {}, dims = {} : ({}) -> {}",
            value.name,
            bracket_list(&dims),
            format_tensor_type(&value.ty),
            format_tensor_type(target)
        ));
        value = Value {
            name,
            ty: target.clone(),
        };
    }
    Ok(value)
}

pub(super) fn lower_binary(
    op: &'static str,
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count(op, inputs, 2)?;
    let lhs = coerce(&inputs[0], output_ty, emitter)?;
    let rhs = coerce(&inputs[1], output_ty, emitter)?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = {op} {}, {} : {}",
        lhs.name,
        rhs.name,
        format_tensor_type(output_ty)
    ));
    Ok(out(name, output_ty))
}

fn promote(a: DType, b: DType) -> DType {
    fn rank(d: DType) -> u8 {
        match d {
            DType::Bool => 0,
            DType::I32 => 1,
            DType::I64 => 2,
            DType::F32 => 3,
            _ => 4,
        }
    }
    if rank(a) >= rank(b) {
        a
    } else {
        b
    }
}

pub(super) fn lower_compare(
    dir: &CompareDir,
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.compare", inputs, 2)?;
    let operand_dtype = promote(inputs[0].ty.dtype, inputs[1].ty.dtype);
    let operand_ty = TensorType::new(output_ty.shape.clone(), operand_dtype, "compare operand")?;
    let lhs = coerce(&inputs[0], &operand_ty, emitter)?;
    let rhs = coerce(&inputs[1], &operand_ty, emitter)?;
    let direction = match dir {
        CompareDir::Eq => "EQ",
        CompareDir::Lt => "LT",
        CompareDir::Le => "LE",
        CompareDir::Gt => "GT",
        CompareDir::Ge => "GE",
    };
    let compare_type = match operand_dtype {
        DType::F32 | DType::F64 => "FLOAT",
        DType::Bool => "UNSIGNED",
        _ => "SIGNED",
    };
    let bool_ty = TensorType::new(output_ty.shape.clone(), DType::Bool, "compare result")?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = \"stablehlo.compare\"({}, {}) {{comparison_direction = #stablehlo<comparison_direction {direction}>, compare_type = #stablehlo<comparison_type {compare_type}>}} : ({}, {}) -> {}",
        lhs.name,
        rhs.name,
        format_tensor_type(&operand_ty),
        format_tensor_type(&operand_ty),
        format_tensor_type(&bool_ty)
    ));
    let value = Value { name, ty: bool_ty };
    coerce(&value, output_ty, emitter)
}

/// Turn an arbitrary-dtype predicate into `i1`.
fn as_predicate(pred: &Value, emitter: &mut Emitter) -> Result<Value> {
    if pred.ty.dtype == DType::Bool {
        return Ok(pred.clone());
    }
    let zero = emit_scalar_constant(pred.ty.dtype, 0.0, emitter)?;
    let zero = coerce(&zero, &pred.ty, emitter)?;
    let compare_type = match pred.ty.dtype {
        DType::F32 | DType::F64 => "FLOAT",
        _ => "SIGNED",
    };
    let bool_ty = TensorType::new(pred.ty.shape.clone(), DType::Bool, "predicate")?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = \"stablehlo.compare\"({}, {}) {{comparison_direction = #stablehlo<comparison_direction NE>, compare_type = #stablehlo<comparison_type {compare_type}>}} : ({}, {}) -> {}",
        pred.name,
        zero.name,
        format_tensor_type(&pred.ty),
        format_tensor_type(&pred.ty),
        format_tensor_type(&bool_ty)
    ));
    Ok(Value { name, ty: bool_ty })
}

pub(super) fn lower_select(
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.select", inputs, 3)?;
    let pred = as_predicate(&inputs[0], emitter)?;
    let pred_ty = TensorType::new(output_ty.shape.clone(), DType::Bool, "select predicate")?;
    let pred = coerce(&pred, &pred_ty, emitter)?;
    let on_true = coerce(&inputs[1], output_ty, emitter)?;
    let on_false = coerce(&inputs[2], output_ty, emitter)?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = \"stablehlo.select\"({}, {}, {}) : ({}, {}, {}) -> {}",
        pred.name,
        on_true.name,
        on_false.name,
        format_tensor_type(&pred_ty),
        format_tensor_type(output_ty),
        format_tensor_type(output_ty),
        format_tensor_type(output_ty)
    ));
    Ok(out(name, output_ty))
}

/// tenferro order is `(x, lower, upper)`; StableHLO is `(min, operand, max)`.
pub(super) fn lower_clamp(
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.clamp", inputs, 3)?;
    let x = coerce(&inputs[0], output_ty, emitter)?;
    let lower = coerce(&inputs[1], output_ty, emitter)?;
    let upper = coerce(&inputs[2], output_ty, emitter)?;
    let ty = format_tensor_type(output_ty);
    let name = emitter.value();
    emitter.line(format!(
        "{name} = \"stablehlo.clamp\"({}, {}, {}) : ({ty}, {ty}, {ty}) -> {ty}",
        lower.name, x.name, upper.name
    ));
    Ok(out(name, output_ty))
}

pub(super) fn lower_slice(
    config: &SliceConfig,
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.slice", inputs, 1)?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = \"stablehlo.slice\"({}) {{start_indices = {}, limit_indices = {}, strides = {}}} : ({}) -> {}",
        inputs[0].name,
        i64_list(&config.starts),
        i64_list(&config.limits),
        i64_list(&config.strides),
        format_tensor_type(&inputs[0].ty),
        format_tensor_type(output_ty)
    ));
    Ok(out(name, output_ty))
}

pub(super) fn lower_pad(
    config: &PadConfig,
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.pad", inputs, 1)?;
    // Default: express zero padding as concatenation with zero blocks. With the
    // zml PJRT CPU plugin (2026-09-08) the sum of two `stablehlo.pad` results
    // came back nondeterministic (uninitialised padding), while each pad alone
    // was fine; concatenation is exact and deterministic on the same plugin.
    // `TENFERRO_XLA_NATIVE_PAD=1` restores the direct `stablehlo.pad` lowering.
    if std::env::var_os("TENFERRO_XLA_NATIVE_PAD").is_none()
        && config.interior_padding.iter().all(|&p| p == 0)
        && config.edge_padding_low.iter().all(|&p| p >= 0)
        && config.edge_padding_high.iter().all(|&p| p >= 0)
    {
        // Zero blocks concatenated axis by axis.
        let zero = emit_scalar_constant(output_ty.dtype, 0.0, emitter)?;
        let mut current = inputs[0].clone();
        for axis in 0..current.ty.shape.len() {
            let low = config.edge_padding_low[axis] as usize;
            let high = config.edge_padding_high[axis] as usize;
            if low == 0 && high == 0 {
                continue;
            }
            let mut parts = Vec::new();
            let block = |extent: usize, emitter: &mut Emitter| -> Result<Value> {
                let mut shape = current.ty.shape.clone();
                shape[axis] = extent;
                let ty = TensorType::new(shape, output_ty.dtype, "pad block")?;
                coerce(&zero, &ty, emitter)
            };
            if low > 0 {
                parts.push(block(low, emitter)?);
            }
            parts.push(current.clone());
            if high > 0 {
                parts.push(block(high, emitter)?);
            }
            let mut shape = current.ty.shape.clone();
            shape[axis] += low + high;
            let ty = TensorType::new(shape, output_ty.dtype, "pad result")?;
            current = lower_concatenate(axis, &parts, &ty, emitter)?;
        }
        return Ok(current);
    }
    let zero = emit_scalar_constant(output_ty.dtype, 0.0, emitter)?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = \"stablehlo.pad\"({}, {}) {{edge_padding_low = {}, edge_padding_high = {}, interior_padding = {}}} : ({}, {}) -> {}",
        inputs[0].name,
        zero.name,
        i64_list(&config.edge_padding_low),
        i64_list(&config.edge_padding_high),
        i64_list(&config.interior_padding),
        format_tensor_type(&inputs[0].ty),
        format_tensor_type(&zero.ty),
        format_tensor_type(output_ty)
    ));
    Ok(out(name, output_ty))
}

pub(super) fn lower_concatenate(
    axis: usize,
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    if inputs.is_empty() {
        return Err(Error::InvalidProgram {
            message: "stablehlo.concatenate needs at least one input".to_string(),
        });
    }
    let mut operands = Vec::with_capacity(inputs.len());
    for input in inputs {
        if input.ty.dtype == output_ty.dtype {
            operands.push(input.clone());
        } else {
            let ty = TensorType::new(input.ty.shape.clone(), output_ty.dtype, "concatenate")?;
            operands.push(coerce(input, &ty, emitter)?);
        }
    }
    let names = operands
        .iter()
        .map(|v| v.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let types = operands
        .iter()
        .map(|v| format_tensor_type(&v.ty))
        .collect::<Vec<_>>()
        .join(", ");
    let name = emitter.value();
    emitter.line(format!(
        "{name} = \"stablehlo.concatenate\"({names}) {{dimension = {axis} : i64}} : ({types}) -> {}",
        format_tensor_type(output_ty)
    ));
    Ok(out(name, output_ty))
}

pub(super) fn lower_reverse(
    axes: &[usize],
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.reverse", inputs, 1)?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = \"stablehlo.reverse\"({}) {{dimensions = {}}} : ({}) -> {}",
        inputs[0].name,
        i64_list(axes),
        format_tensor_type(&inputs[0].ty),
        format_tensor_type(output_ty)
    ));
    Ok(out(name, output_ty))
}

/// StableHLO wants the index vector as a real axis; tenferro (like XLA)
/// accepts `index_vector_dim == rank(indices)` as an implicit trailing axis.
fn explicit_index_vector(
    indices: &Value,
    index_vector_dim: usize,
    emitter: &mut Emitter,
) -> Result<Value> {
    if index_vector_dim < indices.ty.shape.len() {
        return Ok(indices.clone());
    }
    if index_vector_dim != indices.ty.shape.len() {
        return Err(Error::InvalidProgram {
            message: format!(
                "index_vector_dim {index_vector_dim} out of range for indices {:?}",
                indices.ty.shape
            ),
        });
    }
    let mut shape = indices.ty.shape.clone();
    shape.push(1);
    let ty = TensorType::new(shape, indices.ty.dtype, "gather/scatter indices")?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = stablehlo.reshape {} : ({}) -> {}",
        indices.name,
        format_tensor_type(&indices.ty),
        format_tensor_type(&ty)
    ));
    Ok(Value { name, ty })
}

pub(super) fn lower_gather(
    config: &GatherConfig,
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.gather", inputs, 2)?;
    let indices = explicit_index_vector(&inputs[1], config.index_vector_dim, emitter)?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = \"stablehlo.gather\"({}, {}) {{dimension_numbers = #stablehlo.gather<offset_dims = {}, collapsed_slice_dims = {}, start_index_map = {}, index_vector_dim = {}>, indices_are_sorted = false, slice_sizes = {}}} : ({}, {}) -> {}",
        inputs[0].name,
        indices.name,
        bracket_list(&config.offset_dims),
        bracket_list(&config.collapsed_slice_dims),
        bracket_list(&config.start_index_map),
        config.index_vector_dim,
        i64_list(&config.slice_sizes),
        format_tensor_type(&inputs[0].ty),
        format_tensor_type(&indices.ty),
        format_tensor_type(output_ty)
    ));
    Ok(out(name, output_ty))
}

/// tenferro scatter is add-scatter: `operand + scatter_add(indices, updates)`.
pub(super) fn lower_scatter(
    config: &ScatterConfig,
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.scatter", inputs, 3)?;
    let operand = coerce(&inputs[0], output_ty, emitter)?;
    let indices = explicit_index_vector(&inputs[1], config.index_vector_dim, emitter)?;
    let updates = if inputs[2].ty.dtype == output_ty.dtype {
        inputs[2].clone()
    } else {
        let ty = TensorType::new(
            inputs[2].ty.shape.clone(),
            output_ty.dtype,
            "scatter updates",
        )?;
        coerce(&inputs[2], &ty, emitter)?
    };
    let scalar = format_tensor_type(&TensorType::scalar(output_ty.dtype, "scatter body")?);
    let lhs = emitter.value();
    let rhs = emitter.value();
    let sum = emitter.value();
    let name = emitter.value();
    emitter.line(format!(
        "{name} = \"stablehlo.scatter\"({}, {}, {}) ({{",
        operand.name, indices.name, updates.name
    ));
    emitter.line(format!("  ^bb0({lhs}: {scalar}, {rhs}: {scalar}):"));
    emitter.line(format!("    {sum} = stablehlo.add {lhs}, {rhs} : {scalar}"));
    emitter.line(format!("    stablehlo.return {sum} : {scalar}"));
    emitter.line(format!(
        "}}) {{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = {}, inserted_window_dims = {}, scatter_dims_to_operand_dims = {}, index_vector_dim = {}>, indices_are_sorted = false, unique_indices = false}} : ({}, {}, {}) -> {}",
        bracket_list(&config.update_window_dims),
        bracket_list(&config.inserted_window_dims),
        bracket_list(&config.scatter_dims_to_operand_dims),
        config.index_vector_dim,
        format_tensor_type(&operand.ty),
        format_tensor_type(&indices.ty),
        format_tensor_type(&updates.ty),
        format_tensor_type(output_ty)
    ));
    Ok(out(name, output_ty))
}

/// `reduce` with an arbitrary monoid (`stablehlo.multiply/maximum/minimum`).
pub(super) fn lower_reduce(
    body: &'static str,
    init: f64,
    axes: &[usize],
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.reduce", inputs, 1)?;
    let init_ty = TensorType::scalar(output_ty.dtype, "reduce init")?;
    let init_name = emitter.value();
    emitter.line(format!(
        "{init_name} = stablehlo.constant dense<{}> : {}",
        match output_ty.dtype {
            DType::F32 | DType::F64 => super::format_float(init),
            other => scalar_literal(other, init),
        },
        format_tensor_type(&init_ty)
    ));
    let name = emitter.value();
    emitter.line(format!(
        "{name} = stablehlo.reduce({} init: {init_name}) applies {body} across dimensions = {} : ({}, {}) -> {}",
        inputs[0].name,
        bracket_list(axes),
        format_tensor_type(&inputs[0].ty),
        format_tensor_type(&init_ty),
        format_tensor_type(output_ty)
    ));
    Ok(out(name, output_ty))
}
