# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Input validation utilities for TensorFlow.

This module provides secure input validation patterns to prevent common
security vulnerabilities such as injection attacks, path traversal,
and resource exhaustion.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import six

# Lazy tf_export for TensorFlow integration; gracefully handle if not available
def _get_tf_export():
  try:
    from tensorflow.python.util.tf_export import tf_export as _tf_export
    return _tf_export
  except Exception:  # Catch ImportError, AttributeError, etc.
    def tf_export(*args, **kwargs):
      def decorator(func):
        return func
      return decorator
    return tf_export

tf_export = _get_tf_export()


# =============================================================================
# Validation Exceptions
# =============================================================================

class ValidationError(ValueError):
  """Raised when input validation fails."""
  pass


class SecurityValidationError(ValidationError):
  """Raised when input fails security validation (potential attack)."""
  pass


# =============================================================================
# String Validation
# =============================================================================

# Patterns for detecting potential injection attacks
_SQL_INJECTION_PATTERNS = [
    r"(?i)(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
    r"(?i)(--|;|/\*|\*/|@@|@|char\(|nchar\(|varchar\(|nvarchar\()",
    r"(?i)(\b(or|and)\b\s+\d+\s*=\s*\d+)",
    r"(?i)('|\"|;)\s*(or|and)\s*\d+",
]

_COMMAND_INJECTION_PATTERNS = [
    r"[;&|`$(){}\[\]<>]",
    r"\$\(",
    r"`.*`",
    r"\|\s*\w+",
    r">\s*/dev/",
    r"<\s*/dev/",
]

_PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e%2f",
    r"%2e%2e%5c",
    r"\.\.%2f",
    r"\.\.%5c",
]

_LDAP_INJECTION_PATTERNS = [
    r"[()*\\]",
    r"\x00",
]

_XSS_PATTERNS = [
    r"(?i)<script\b[^>]*>.*?</script>",
    r"(?i)javascript:",
    r"(?i)on\w+\s*=",
    r"(?i)<iframe\b",
    r"(?i)<object\b",
    r"(?i)<embed\b",
]


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
  """Compile regex patterns with case-insensitive flag where appropriate."""
  compiled = []
  for pattern in patterns:
    try:
      compiled.append(re.compile(pattern))
    except re.error:
      # Fallback for invalid patterns
      compiled.append(re.compile(re.escape(pattern)))
  return compiled


_SQL_INJECTION_REGEX = _compile_patterns(_SQL_INJECTION_PATTERNS)
_COMMAND_INJECTION_REGEX = _compile_patterns(_COMMAND_INJECTION_PATTERNS)
_PATH_TRAVERSAL_REGEX = _compile_patterns(_PATH_TRAVERSAL_PATTERNS)
_LDAP_INJECTION_REGEX = _compile_patterns(_LDAP_INJECTION_PATTERNS)
_XSS_REGEX = _compile_patterns(_XSS_PATTERNS)


@tf_export("validation.validate_string")
def validate_string(
    value: Any,
    param_name: str = "input",
    min_length: int = 0,
    max_length: int = 10000,
    allow_empty: bool = False,
    allowed_chars: Optional[str] = None,
    forbidden_patterns: Optional[List[re.Pattern]] = None,
    strip_whitespace: bool = True,
) -> str:
  """Validate and sanitize a string input.

  Args:
    value: The value to validate.
    param_name: Name of the parameter (for error messages).
    min_length: Minimum allowed length after stripping.
    max_length: Maximum allowed length.
    allow_empty: Whether empty strings are allowed.
    allowed_chars: Regex pattern of allowed characters (e.g., r'^[a-zA-Z0-9_-]+$').
    forbidden_patterns: Additional regex patterns that must not match.
    strip_whitespace: Whether to strip leading/trailing whitespace.

  Returns:
    The validated (and possibly stripped) string.

  Raises:
    ValidationError: If validation fails.
    SecurityValidationError: If security patterns are detected.
  """
  if value is None:
    if allow_empty:
      return ""
    raise ValidationError(f"{param_name}: value cannot be None")

  if not isinstance(value, (six.string_types, bytes)):
    raise ValidationError(f"{param_name}: expected string, got {type(value).__name__}")

  if isinstance(value, bytes):
    try:
      value = value.decode("utf-8")
    except UnicodeDecodeError:
      raise ValidationError(f"{param_name}: invalid UTF-8 sequence")

  if strip_whitespace:
    value = value.strip()

  if not value:
    if allow_empty:
      return ""
    raise ValidationError(f"{param_name}: value cannot be empty")

  if len(value) < min_length:
    raise ValidationError(
        f"{param_name}: length {len(value)} < minimum {min_length}")

  if len(value) > max_length:
    raise ValidationError(
        f"{param_name}: length {len(value)} > maximum {max_length}")

  # Check for security-related patterns
  _check_security_patterns(value, param_name)

  # Check custom forbidden patterns
  if forbidden_patterns:
    for pattern in forbidden_patterns:
      if pattern.search(value):
        raise ValidationError(
            f"{param_name}: value matches forbidden pattern")

  # Check allowed characters
  if allowed_chars:
    allowed_regex = re.compile(allowed_chars)
    if not allowed_regex.match(value):
      raise ValidationError(
          f"{param_name}: value contains disallowed characters")

  return value


def _check_security_patterns(value: str, param_name: str) -> None:
  """Check for common injection attack patterns."""
  # Check SQL injection
  for pattern in _SQL_INJECTION_REGEX:
    if pattern.search(value):
      raise SecurityValidationError(
          f"{param_name}: potential SQL injection detected")

  # Check command injection
  for pattern in _COMMAND_INJECTION_REGEX:
    if pattern.search(value):
      raise SecurityValidationError(
          f"{param_name}: potential command injection detected")

  # Check path traversal
  for pattern in _PATH_TRAVERSAL_REGEX:
    if pattern.search(value):
      raise SecurityValidationError(
          f"{param_name}: potential path traversal detected")

  # Check LDAP injection
  for pattern in _LDAP_INJECTION_REGEX:
    if pattern.search(value):
      raise SecurityValidationError(
          f"{param_name}: potential LDAP injection detected")

  # Check XSS
  for pattern in _XSS_REGEX:
    if pattern.search(value):
      raise SecurityValidationError(
          f"{param_name}: potential XSS detected")


@tf_export("validation.validate_filename")
def validate_filename(
    filename: Any,
    param_name: str = "filename",
    allowed_extensions: Optional[List[str]] = None,
    max_length: int = 255,
) -> str:
  """Validate a filename for security.

  Args:
    filename: The filename to validate.
    param_name: Name of the parameter.
    allowed_extensions: List of allowed extensions (e.g., ['.txt', '.csv']).
    max_length: Maximum filename length.

  Returns:
    The validated filename.

  Raises:
    ValidationError: If validation fails.
    SecurityValidationError: If path traversal is detected.
  """
  filename = validate_string(
      filename,
      param_name=param_name,
      min_length=1,
      max_length=max_length,
      allowed_chars=r"^[a-zA-Z0-9._\- ]+$",
  )

  # Check for path traversal attempts
  if os.path.isabs(filename):
    raise SecurityValidationError(
        f"{param_name}: absolute paths not allowed")

  if ".." in filename or filename.startswith("."):
    raise SecurityValidationError(
        f"{param_name}: path traversal detected")

  # Check extension
  if allowed_extensions:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [e.lower() for e in allowed_extensions]:
      raise ValidationError(
          f"{param_name}: extension '{ext}' not in allowed list: {allowed_extensions}")

  return filename


@tf_export("validation.validate_path")
def validate_path(
    path: Any,
    param_name: str = "path",
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    allowed_base_dirs: Optional[List[str]] = None,
) -> str:
  """Validate a filesystem path for security.

  Args:
    path: The path to validate.
    param_name: Name of the parameter.
    must_exist: Whether the path must exist.
    must_be_file: Whether the path must be a file.
    must_be_dir: Whether the path must be a directory.
    allowed_base_dirs: List of allowed base directories (prevents traversal).

  Returns:
    The validated absolute path.

  Raises:
    ValidationError: If validation fails.
    SecurityValidationError: If path traversal is detected.
  """
  path = validate_string(path, param_name=param_name, min_length=1)

  # Resolve to absolute path
  abs_path = os.path.abspath(path)

  # Check for path traversal
  if ".." in path:
    raise SecurityValidationError(
        f"{param_name}: path traversal detected")

  # Check against allowed base directories
  if allowed_base_dirs:
    allowed = False
    for base_dir in allowed_base_dirs:
      base_abs = os.path.abspath(base_dir)
      if abs_path.startswith(base_abs + os.sep) or abs_path == base_abs:
        allowed = True
        break
    if not allowed:
      raise SecurityValidationError(
          f"{param_name}: path outside allowed directories")

  # Check existence
  if must_exist and not os.path.exists(abs_path):
    raise ValidationError(f"{param_name}: path does not exist: {abs_path}")

  if must_be_file and not os.path.isfile(abs_path):
    raise ValidationError(f"{param_name}: not a file: {abs_path}")

  if must_be_dir and not os.path.isdir(abs_path):
    raise ValidationError(f"{param_name}: not a directory: {abs_path}")

  return abs_path


# =============================================================================
# Numeric Validation
# =============================================================================

@tf_export("validation.validate_int")
def validate_int(
    value: Any,
    param_name: str = "value",
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    allow_none: bool = False,
) -> Optional[int]:
  """Validate an integer value.

  Args:
    value: The value to validate.
    param_name: Name of the parameter.
    min_value: Minimum allowed value (inclusive).
    max_value: Maximum allowed value (inclusive).
    allow_none: Whether None is allowed.

  Returns:
    The validated integer.

  Raises:
    ValidationError: If validation fails.
  """
  if value is None:
    if allow_none:
      return None
    raise ValidationError(f"{param_name}: value cannot be None")

  # Reject booleans (bool is subclass of int in Python)
  if isinstance(value, bool):
    raise ValidationError(f"{param_name}: expected int, got bool")

  try:
    result = int(value)
  except (TypeError, ValueError):
    raise ValidationError(f"{param_name}: expected int, got {type(value).__name__}")

  if min_value is not None and result < min_value:
    raise ValidationError(
        f"{param_name}: value {result} < minimum {min_value}")

  if max_value is not None and result > max_value:
    raise ValidationError(
        f"{param_name}: value {result} > maximum {max_value}")

  return result


@tf_export("validation.validate_float")
def validate_float(
    value: Any,
    param_name: str = "value",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
    allow_none: bool = False,
) -> Optional[float]:
  """Validate a float value.

  Args:
    value: The value to validate.
    param_name: Name of the parameter.
    min_value: Minimum allowed value (inclusive).
    max_value: Maximum allowed value (inclusive).
    allow_nan: Whether NaN is allowed.
    allow_inf: Whether infinity is allowed.
    allow_none: Whether None is allowed.

  Returns:
    The validated float.

  Raises:
    ValidationError: If validation fails.
  """
  if value is None:
    if allow_none:
      return None
    raise ValidationError(f"{param_name}: value cannot be None")

  # Reject booleans
  if isinstance(value, bool):
    raise ValidationError(f"{param_name}: expected float, got bool")

  try:
    result = float(value)
  except (TypeError, ValueError):
    raise ValidationError(f"{param_name}: expected float, got {type(value).__name__}")

  if not allow_nan and np.isnan(result):
    raise ValidationError(f"{param_name}: NaN not allowed")

  if not allow_inf and np.isinf(result):
    raise ValidationError(f"{param_name}: infinity not allowed")

  if min_value is not None and result < min_value:
    raise ValidationError(
        f"{param_name}: value {result} < minimum {min_value}")

  if max_value is not None and result > max_value:
    raise ValidationError(
        f"{param_name}: value {result} > maximum {max_value}")

  return result


# =============================================================================
# Array/Tensor Validation
# =============================================================================

@tf_export("validation.validate_shape")
def validate_shape(
    tensor: Any,
    expected_shape: Sequence[Optional[int]],
    param_name: str = "tensor",
) -> Tuple[int, ...]:
  """Validate tensor/array shape.

  Args:
    tensor: Tensor or array to validate.
    expected_shape: Expected shape tuple (None for any dimension).
    param_name: Name of the parameter.

  Returns:
    The actual shape as a tuple.

  Raises:
    ValidationError: If shape doesn't match.
  """
  if hasattr(tensor, "shape"):
    actual_shape = tensor.shape
  elif hasattr(tensor, "get_shape"):
    actual_shape = tensor.get_shape().as_list()
  elif isinstance(tensor, np.ndarray):
    actual_shape = tensor.shape
  else:
    raise ValidationError(f"{param_name}: expected tensor/array, got {type(tensor).__name__}")

  actual = tuple(actual_shape)
  expected = tuple(expected_shape)

  if len(actual) != len(expected):
    raise ValidationError(
        f"{param_name}: rank {len(actual)} != expected {len(expected)}")

  for i, (a, e) in enumerate(zip(actual, expected)):
    if e is not None and a != e:
      raise ValidationError(
          f"{param_name}: dimension {i}: {a} != expected {e}")

  return actual


@tf_export("validation.validate_dtype")
def validate_dtype(
    tensor: Any,
    allowed_dtypes: Sequence[Any],
    param_name: str = "tensor",
) -> Any:
  """Validate tensor/array dtype.

  Args:
    tensor: Tensor or array to validate.
    allowed_dtypes: Sequence of allowed dtypes.
    param_name: Name of the parameter.

  Returns:
    The actual dtype.

  Raises:
    ValidationError: If dtype not in allowed list.
  """
  if hasattr(tensor, "dtype"):
    actual_dtype = tensor.dtype
  elif isinstance(tensor, np.ndarray):
    actual_dtype = tensor.dtype
  else:
    raise ValidationError(f"{param_name}: expected tensor/array, got {type(tensor).__name__}")

  if actual_dtype not in allowed_dtypes:
    raise ValidationError(
        f"{param_name}: dtype {actual_dtype} not in allowed: {allowed_dtypes}")

  return actual_dtype


@tf_export("validation.validate_tensor")
def validate_tensor(
    tensor: Any,
    param_name: str = "tensor",
    expected_shape: Optional[Sequence[Optional[int]]] = None,
    allowed_dtypes: Optional[Sequence[Any]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> Any:
  """Comprehensive tensor validation.

  Args:
    tensor: Tensor to validate.
    param_name: Name of the parameter.
    expected_shape: Expected shape (None for any dimension).
    allowed_dtypes: Allowed dtypes.
    min_value: Minimum element value.
    max_value: Maximum element value.
    allow_nan: Whether NaN is allowed.
    allow_inf: Whether infinity is allowed.

  Returns:
    The validated tensor.

  Raises:
    ValidationError: If validation fails.
  """
  if tensor is None:
    raise ValidationError(f"{param_name}: tensor cannot be None")

  # Convert to numpy for validation if needed
  if hasattr(tensor, "numpy"):
    np_tensor = tensor.numpy()
  elif isinstance(tensor, np.ndarray):
    np_tensor = tensor
  else:
    try:
      np_tensor = np.asarray(tensor)
    except Exception:
      raise ValidationError(f"{param_name}: cannot convert to array")

  # Validate shape
  if expected_shape is not None:
    validate_shape(np_tensor, expected_shape, param_name)

  # Validate dtype
  if allowed_dtypes is not None:
    validate_dtype(np_tensor, allowed_dtypes, param_name)

  # Validate value range
  if min_value is not None:
    actual_min = np.min(np_tensor)
    if actual_min < min_value:
      raise ValidationError(
          f"{param_name}: min value {actual_min} < {min_value}")

  if max_value is not None:
    actual_max = np.max(np_tensor)
    if actual_max > max_value:
      raise ValidationError(
          f"{param_name}: max value {actual_max} > {max_value}")

  # Check for NaN/Inf
  if not allow_nan and np.any(np.isnan(np_tensor)):
    raise ValidationError(f"{param_name}: contains NaN values")

  if not allow_inf and np.any(np.isinf(np_tensor)):
    raise ValidationError(f"{param_name}: contains Inf values")

  return tensor


# =============================================================================
# Collection Validation
# =============================================================================

@tf_export("validation.validate_list")
def validate_list(
    value: Any,
    param_name: str = "list",
    element_validator: Optional[Callable[[Any], Any]] = None,
    min_length: int = 0,
    max_length: Optional[int] = None,
    allow_none: bool = False,
    unique: bool = False,
) -> List[Any]:
  """Validate a list/sequence.

  Args:
    value: The value to validate.
    param_name: Name of the parameter.
    element_validator: Callable to validate each element.
    min_length: Minimum length.
    max_length: Maximum length.
    allow_none: Whether None is allowed.
    unique: Whether elements must be unique.

  Returns:
    The validated list.

  Raises:
    ValidationError: If validation fails.
  """
  if value is None:
    if allow_none:
      return []
    raise ValidationError(f"{param_name}: value cannot be None")

  if not isinstance(value, (list, tuple)):
    raise ValidationError(f"{param_name}: expected list/tuple, got {type(value).__name__}")

  result = list(value)

  if len(result) < min_length:
    raise ValidationError(
        f"{param_name}: length {len(result)} < minimum {min_length}")

  if max_length is not None and len(result) > max_length:
    raise ValidationError(
        f"{param_name}: length {len(result)} > maximum {max_length}")

  if unique and len(result) != len(set(result)):
    raise ValidationError(f"{param_name}: elements must be unique")

  if element_validator:
    validated = []
    for i, elem in enumerate(result):
      try:
        validated.append(element_validator(elem))
      except ValidationError as e:
        raise ValidationError(f"{param_name}[{i}]: {e}")
    result = validated

  return result


@tf_export("validation.validate_dict")
def validate_dict(
    value: Any,
    param_name: str = "dict",
    required_keys: Optional[List[str]] = None,
    allowed_keys: Optional[List[str]] = None,
    key_validator: Optional[Callable[[Any], Any]] = None,
    value_validator: Optional[Callable[[Any], Any]] = None,
    allow_none: bool = False,
) -> dict:
  """Validate a dictionary.

  Args:
    value: The value to validate.
    param_name: Name of the parameter.
    required_keys: Keys that must be present.
    allowed_keys: Keys that are allowed (None = any).
    key_validator: Callable to validate keys.
    value_validator: Callable to validate values.
    allow_none: Whether None is allowed.

  Returns:
    The validated dictionary.

  Raises:
    ValidationError: If validation fails.
  """
  if value is None:
    if allow_none:
      return {}
    raise ValidationError(f"{param_name}: value cannot be None")

  if not isinstance(value, dict):
    raise ValidationError(f"{param_name}: expected dict, got {type(value).__name__}")

  result = {}

  # Check required keys
  if required_keys:
    for key in required_keys:
      if key not in value:
        raise ValidationError(f"{param_name}: missing required key '{key}'")

  # Validate each key-value pair
  for key, val in value.items():
    # Validate key
    if allowed_keys is not None and key not in allowed_keys:
      raise ValidationError(f"{param_name}: key '{key}' not allowed")

    if key_validator:
      try:
        key = key_validator(key)
      except ValidationError as e:
        raise ValidationError(f"{param_name}.{key}: {e}")

    # Validate value
    if value_validator:
      try:
        val = value_validator(val)
      except ValidationError as e:
        raise ValidationError(f"{param_name}['{key}']: {e}")

    result[key] = val

  return result


# =============================================================================
# Composite Validators
# =============================================================================

@tf_export("validation.validate_config")
def validate_config(
    config: Any,
    schema: dict,
    param_name: str = "config",
) -> dict:
  """Validate a configuration dictionary against a schema.

  Args:
    config: Configuration dictionary to validate.
    schema: Schema dict mapping keys to validator functions or specs.
    param_name: Name of the parameter.

  Returns:
    Validated configuration dictionary.

  Raises:
    ValidationError: If validation fails.
  """
  if not isinstance(config, dict):
    raise ValidationError(f"{param_name}: expected dict, got {type(config).__name__}")

  result = {}

  for key, spec in schema.items():
    required = True
    validator = None
    default = None

    if isinstance(spec, dict):
      required = spec.get("required", True)
      validator = spec.get("validator")
      default = spec.get("default")
    else:
      validator = spec

    if key not in config:
      if required:
        raise ValidationError(f"{param_name}: missing required key '{key}'")
      if default is not None:
        result[key] = default
      continue

    value = config[key]

    if validator:
      try:
        result[key] = validator(value)
      except ValidationError as e:
        raise ValidationError(f"{param_name}.{key}: {e}")
    else:
      result[key] = value

  # Check for unexpected keys
  unexpected = set(config.keys()) - set(schema.keys())
  if unexpected:
    raise ValidationError(f"{param_name}: unexpected keys: {unexpected}")

  return result


# =============================================================================
# Sanitization Utilities
# =============================================================================

@tf_export("validation.sanitize_string")
def sanitize_string(
    value: str,
    max_length: int = 10000,
    allow_unicode: bool = True,
) -> str:
  """Sanitize a string for safe display/storage.

  Args:
    value: String to sanitize.
    max_length: Maximum length.
    allow_unicode: Whether to allow Unicode characters.

  Returns:
    Sanitized string.
  """
  if not isinstance(value, str):
    value = str(value)

  # Remove control characters except newline, tab
  if allow_unicode:
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
  else:
    value = re.sub(r"[^\x20-\x7e\n\t]", "", value)

  # Truncate
  if len(value) > max_length:
    value = value[:max_length]

  return value


@tf_export("validation.sanitize_path")
def sanitize_path(path: str, base_dir: str) -> str:
  """Sanitize a path to be within a base directory.

  Args:
    path: Path to sanitize.
    base_dir: Base directory to restrict to.

  Returns:
    Sanitized absolute path.

  Raises:
    SecurityValidationError: If path escapes base directory.
  """
  path = validate_string(path, min_length=1)
  base_dir = os.path.abspath(base_dir)

  # Resolve path
  abs_path = os.path.abspath(os.path.join(base_dir, path))

  # Ensure it's within base_dir
  if not abs_path.startswith(base_dir + os.sep) and abs_path != base_dir:
    raise SecurityValidationError("Path traversal attempt detected")

  return abs_path


# =============================================================================
# Decorator for Function Validation
# =============================================================================

def validate_args(**validators):
  """Decorator to validate function arguments.

  Usage:
    @validate_args(
        x=validate_int(min_value=0, max_value=100),
        name=validate_string(max_length=50),
    )
    def my_function(x, name):
        ...
  """
  def decorator(func):
    def wrapper(*args, **kwargs):
      import inspect
      sig = inspect.signature(func)
      bound = sig.bind(*args, **kwargs)
      bound.apply_defaults()

      for param_name, validator in validators.items():
        if param_name in bound.arguments:
          try:
            bound.arguments[param_name] = validator(bound.arguments[param_name])
          except ValidationError as e:
            raise ValidationError(f"Argument '{param_name}': {e}")

      return func(*bound.args, **bound.kwargs)
    return wrapper
  return decorator


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ValidationError",
    "SecurityValidationError",
    "validate_string",
    "validate_filename",
    "validate_path",
    "validate_int",
    "validate_float",
    "validate_shape",
    "validate_dtype",
    "validate_tensor",
    "validate_list",
    "validate_dict",
    "validate_config",
    "sanitize_string",
    "sanitize_path",
    "validate_args",
]