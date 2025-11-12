## Configuration Guidelines

- Store canonical experiment settings in YAML files within this directory. Use descriptive filenames (e.g., `config_teacher_baseline.yaml`, `config_student_kd.yaml`) when you branch out from the default `config.yaml`.
- Track changes between configurations via git commits or pull requests. Include short rationales in commit messages or a changelog so experiment provenance remains clear.
- Keep secret credentials (e.g., Weights & Biases API keys) out of these files. Instead, reference environment variables and document the expected variable names in the root `README.md`.
- When a configuration depends on generated assets (processed datasets, pretrained checkpoints), document the reproduction steps in the `data/README.md` or dedicated experiment notes.
