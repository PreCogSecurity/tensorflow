# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

**Note:** This repository contains TensorFlow v0.8.0 (legacy). For current TensorFlow security policies, see [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/security/policy).

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security vulnerabilities via:

1. **GitHub Security Advisories** (Preferred):
   - Go to the [Security tab](https://github.com/PreCogSecurity/tensorflow/security/advisories)
   - Click "Report a vulnerability"
   - Fill in the details

2. **Email** (Alternative):
   - Send details to: security@precogsecurity.com
   - Include "SECURITY" in the subject line

### What to Include

Please provide as much detail as possible:

- **Description**: Clear description of the vulnerability
- **Impact**: Potential impact (data exposure, RCE, DoS, etc.)
- **Reproduction**: Steps to reproduce (code, commands, etc.)
- **Affected Versions**: Which versions/branches are affected
- **Suggested Fix**: If you have a fix or mitigation suggestion
- **Your Contact Info**: For follow-up (optional if using GitHub Security Advisories)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Fix Development**: Within 30 days (critical), 90 days (high/medium)
- **Disclosure**: Coordinated with reporter after fix is released

## Security Best Practices for Users

### Environment Configuration

- **Never commit secrets** to version control (API keys, passwords, tokens)
- Use `.env` files (see `.env.example`) for local development
- In production, use secret managers (HashiCorp Vault, AWS Secrets Manager, etc.)
- Rotate credentials regularly

### TensorFlow-Specific Security

1. **Model Loading**: Only load models from trusted sources
   ```python
   # UNSAFE - arbitrary code execution possible
   model = tf.keras.models.load_model(untrusted_path)

   # SAFE - verify source first
   model = tf.keras.models.load_model(verified_path)
   ```

2. **Pickle/Serialization**: Avoid `pickle` for untrusted data
   ```python
   # UNSAFE
   import pickle
   model = pickle.load(untrusted_file)

   # SAFE - use TensorFlow's SavedModel format
   model = tf.saved_model.load(trusted_path)
   ```

3. **Input Validation**: Validate all external inputs
   ```python
   from tensorflow.python.validation import validate_tensor

   # Validate shape, dtype, and value ranges
   validated = validate_tensor(
       user_input,
       expected_shape=(None, 224, 224, 3),
       allowed_dtypes=[tf.float32],
       min_value=0.0,
       max_value=1.0,
   )
   ```

4. **GPU/Resource Limits**: Set resource limits in production
   ```python
   # Limit GPU memory growth
   gpus = tf.config.experimental.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   ```

### Container Security

- Use the provided `Dockerfile` with non-root user
- Scan images regularly: `docker scan tensorflow:latest`
- Keep base images updated (Dependabot handles this)
- Run with minimal privileges: `--user tensorflow --cap-drop=ALL`

### Dependency Management

- Dependabot automatically creates PRs for dependency updates
- Review and merge security updates promptly
- Run `pip-audit` and `safety check` regularly
- Pin dependencies in production requirements

## Secure Development Guidelines

### For Contributors

1. **Code Review**: All changes require security review
2. **Static Analysis**: Run `pylint`, `bandit`, `mypy` before PR
3. **Secrets**: Never commit secrets (pre-commit hooks enforce this)
4. **Dependencies**: Minimize new dependencies; justify each addition

### Input Validation Patterns

Use the validation module (`tensorflow.python.validation`):

```python
from tensorflow.python.validation import (
    validate_string,
    validate_int,
    validate_tensor,
    SecurityValidationError,
)

def process_user_input(user_data):
    # Validate string inputs
    name = validate_string(user_data.get("name"), max_length=100)

    # Validate numeric inputs
    age = validate_int(user_data.get("age"), min_value=0, max_value=150)

    # Validate tensor inputs
    image = validate_tensor(
        user_data.get("image"),
        expected_shape=(None, 224, 224, 3),
        allowed_dtypes=[tf.float32],
        min_value=0.0,
        max_value=1.0,
    )
    return process(name, age, image)
```

### Common Vulnerability Patterns to Avoid

| Pattern | Risk | Mitigation |
|---------|------|------------|
| `eval()` / `exec()` | Code injection | Use `ast.literal_eval()` or avoid |
| `pickle.load()` | RCE | Use `tf.saved_model.load()` |
| `subprocess.shell=True` | Command injection | Use `shell=False`, pass args as list |
| String formatting in SQL | SQL injection | Use parameterized queries |
| `os.path.join` with user input | Path traversal | Validate with `validate_path()` |
| `yaml.load()` without Loader | Code execution | Use `yaml.safe_load()` |

## Security Tooling

### Pre-commit Hooks

Install pre-commit hooks to catch issues early:

```bash
pip install pre-commit
pre-commit install
```

### CI Security Checks

Our CI pipeline runs:
- **Bandit**: Python security linter
- **Safety**: Known vulnerability database check
- **pip-audit**: PyPI vulnerability scan
- **TruffleHog**: Secret scanning
- **GitLeaks**: Additional secret detection
- **Dependency Review**: PR dependency analysis

### Local Security Scanning

```bash
# Run all security checks
bandit -r tensorflow/python
safety check
pip-audit
trufflehog filesystem .
gitleaks detect --source .
```

## Vulnerability Disclosure Process

1. **Report Received** → Acknowledged within 48 hours
2. **Triage** → Severity assessed (CVSS scoring)
3. **Fix Development** → Patch developed and tested
4. **Coordinated Disclosure** → Reporter notified of fix timeline
5. **Release** → Security advisory published with fix
6. **Post-Release** → CVE requested if applicable

## Security Contacts

- **Primary**: security@precogsecurity.com
- **GitHub**: [@precogsecurity](https://github.com/precogsecurity)
- **PGP Key**: Available on request for sensitive communications

## Acknowledgments

We thank all security researchers who responsibly disclose vulnerabilities.

---

**Last Updated**: 2026-10-10
**Policy Version**: 1.0