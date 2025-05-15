#!/usr/bin/env python3

import argparse

from xdsl.parser import Parser
from xdsl.context import Context
from xdsl.dialects.builtin import StringAttr, i32


def convert_mlir_19_to_20(input_file, output_file):
    context = Context(allow_unregistered=True)
    # read file:
    with open(input_file, "r") as infile:
        module = Parser(context, infile.read()).parse_module()

    for op in module.walk():
        if "op_name__" not in op.attributes:
            continue
        assert isinstance(op.attributes["op_name__"], StringAttr)

        op_name = op.attributes["op_name__"].data

        # add acc_type attribute to tosa.conv2d operations
        if op_name in ("tosa.conv2d", "tosa.depthwise_conv2d"):
            op.properties["acc_type"] = i32

        # remove shift from tosa.mul operations
        if op_name == "tosa.mul":
            if "shift" in op.properties:
                del op.properties["shift"]

    with open(output_file, "w") as outfile:
        outfile.write(str(module))


def main():
    parser = argparse.ArgumentParser(
        description="Convert MLIR 19 file to MLIR 20 format."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the input file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output file")

    args = parser.parse_args()

    convert_mlir_19_to_20(args.input, args.output)


if __name__ == "__main__":
    main()
