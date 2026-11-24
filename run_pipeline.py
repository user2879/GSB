from quant_pipeline.cli import main


raise SystemExit(main(["run", *__import__("sys").argv[1:]]))

