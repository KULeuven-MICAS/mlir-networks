import numpy as np
import json
import tensorflow as tf
import argparse


def generate_data(model_path):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Set fixed seed
    np.random.seed(0)

    # Get input details and generate random input
    input_details = interpreter.get_input_details()

    input_data = None

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

        input_data = rand_input
        interpreter.set_tensor(detail["index"], rand_input)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    return input_data, output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect inputs, outputs, and weights of a TFLite model."
    )
    parser.add_argument("model", type=str, help="Path to the .tflite model file")
    parser.add_argument("output_file", type=str, help="Path to the .json output file")
    args = parser.parse_args()
    input, output = generate_data(args.model)
    with open(args.output_file, "w") as f:
        json.dump(
            {
                "input": input.tolist()[0],
                "output": input.tolist()[0],
            },
            f,
        )
