from feedback_pe.generator import generate_project_files


if __name__ == "__main__":
    paths = generate_project_files(".")
    for name, path in paths.items():
        print(f"generated {name}: {path}")
