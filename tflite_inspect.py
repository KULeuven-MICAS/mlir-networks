import numpy as np
import tensorflow as tf
import argparse


def inspect_tflite_model(model_path):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Set fixed seed
    np.random.seed(0)

    # Get input details and generate random input
    input_details = interpreter.get_input_details()
    input_data = []

    print("\n--- Model Inputs ---")
    for i, detail in enumerate(input_details):
        shape = detail["shape"]
        dtype = detail["dtype"]

        if np.issubdtype(dtype, np.floating):
            rand_input = np.random.rand(*shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            rand_input = np.random.randint(
                info.min, info.max + 1, size=shape, dtype=dtype
            )
        else:
            raise ValueError(f"Unsupported input dtype: {dtype}")

        input_data.append(rand_input)
        interpreter.set_tensor(detail["index"], rand_input)

        print(f"Input {i}: name={detail['name']}, shape={shape}, dtype={dtype}")
        print(rand_input, "\n")

    # Run inference
    interpreter.invoke()

    # Get all tensors
    tensor_details = interpreter.get_tensor_details()

    print("\n--- Intermediate and Output Tensors ---")
    for tensor in tensor_details:
        try:
            data = interpreter.get_tensor(tensor["index"])
            print(
                f"{tensor['name']} (index {tensor['index']}): shape={data.shape}, dtype={data.dtype}"
            )
            print(data, "\n")
        except ValueError:
            print(f"{tensor['name']} (index {tensor['index']}): <unavailable>\n")

    print("\n--- Constant Tensors (Weights and Biases) ---")
    input_names = {d["name"] for d in input_details}
    for tensor in tensor_details:
        if tensor["name"] in input_names:
            continue
        if tensor["shape_signature"].size == 0:
            continue  # scalar or unknown
        try:
            data = interpreter.get_tensor(tensor["index"])
            if not np.any(np.isnan(data)) and data.size < 1e6:
                print(
                    f"{tensor['name']} (index {tensor['index']}): shape={data.shape}, dtype={data.dtype}"
                )
                print(data, "\n")
        except ValueError:
            print(f"✗ {tensor['name']} (not accessible)")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect inputs, outputs, and weights of a TFLite model."
    )
    parser.add_argument("model", type=str, help="Path to the .tflite model file")
    args = parser.parse_args()
    inspect_tflite_model(args.model)
