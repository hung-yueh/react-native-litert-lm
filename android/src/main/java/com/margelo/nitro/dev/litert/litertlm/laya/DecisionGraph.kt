package com.margelo.nitro.dev.litert.litertlm.laya

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

/**
 * One compiled decision graph with its input and output buffers addressed by signature name.
 * On the GPU every graph computes in explicit FP32 precision.
 */
internal class DecisionGraph private constructor(
  private val model: CompiledModel,
  private val inputs: Map<String, TensorBuffer>,
  private val outputs: Map<String, TensorBuffer>,
) : AutoCloseable {

  val inputNames: List<String>
    get() = inputs.keys.toList()

  fun write(name: String, values: FloatArray) {
    inputs.getValue(name).writeFloat(values)
  }

  /** Runs the graph; the readback in [read] waits for completion. */
  fun run() {
    model.run(inputs, outputs, SIGNATURE)
  }

  fun read(name: String): FloatArray = outputs.getValue(name).readFloat()

  @Volatile private var closed = false

  /** Releases the buffers, then the model. Safe to call more than once. */
  override fun close() {
    if (closed) return
    closed = true
    try {
      (inputs.values + outputs.values).forEach { runCatching { it.close() } }
    } finally {
      model.close()
    }
  }

  companion object {
    private const val SIGNATURE = "serving_default"
    private const val DEFAULT_CPU_THREADS = 4

    /**
     * Compiles [file] for [accelerator] and allocates the named buffers.
     */
    fun create(
      file: File,
      accelerator: Accelerator,
      inputNames: List<String>,
      outputNames: List<String>,
      numThreads: Int = DEFAULT_CPU_THREADS,
    ): DecisionGraph {
      check(file.isFile) { "Model file not found: ${file.name}" }
      val options =
        CompiledModel.Options(accelerator).apply {
          if (accelerator == Accelerator.GPU) {
            gpuOptions =
              CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
          } else {
            cpuOptions = CompiledModel.CpuOptions(numThreads = numThreads)
          }
        }
      val model = CompiledModel.create(file.absolutePath, options, null)
      val inputs = linkedMapOf<String, TensorBuffer>()
      val outputs = linkedMapOf<String, TensorBuffer>()
      try {
        inputNames.forEach { inputs[it] = model.createInputBuffer(it, SIGNATURE) }
        outputNames.forEach { outputs[it] = model.createOutputBuffer(it, SIGNATURE) }
        return DecisionGraph(model, inputs, outputs)
      } catch (failure: Throwable) {
        (inputs.values + outputs.values).forEach { runCatching { it.close() } }
        model.close()
        throw failure
      }
    }
  }
}
