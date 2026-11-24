from quant_pipeline.cli import main


raise SystemExit(main(["generate-samples", *__import__("sys").argv[1:]]))

