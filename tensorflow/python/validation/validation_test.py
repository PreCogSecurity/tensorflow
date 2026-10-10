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
"""Tests for input validation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.python.platform import googletest
from tensorflow.python.validation import (
    ValidationError,
    SecurityValidationError,
    validate_string,
    validate_filename,
    validate_path,
    validate_int,
    validate_float,
    validate_shape,
    validate_dtype,
    validate_tensor,
    validate_list,
    validate_dict,
    validate_config,
    sanitize_string,
    sanitize_path,
    validate_args,
)


class ValidationTest(googletest.TestCase):

  # -------------------------------------------------------------------------
  # String Validation Tests
  # -------------------------------------------------------------------------

  def test_validate_string_basic(self):
    result = validate_string("hello", "test")
    self.assertEqual(result, "hello")

  def test_validate_string_strips_whitespace(self):
    result = validate_string("  hello  ", "test")
    self.assertEqual(result, "hello")

  def test_validate_string_empty_not_allowed(self):
    with self.assertRaises(ValidationError):
      validate_string("", "test")

  def test_validate_string_empty_allowed(self):
    result = validate_string("", "test", allow_empty=True)
    self.assertEqual(result, "")

  def test_validate_string_none_not_allowed(self):
    with self.assertRaises(ValidationError):
      validate_string(None, "test")

  def test_validate_string_none_allowed(self):
    result = validate_string(None, "test", allow_empty=True)
    self.assertEqual(result, "")

  def test_validate_string_min_length(self):
    with self.assertRaises(ValidationError):
      validate_string("hi", "test", min_length=5)

  def test_validate_string_max_length(self):
    with self.assertRaises(ValidationError):
      validate_string("hello world", "test", max_length=5)

  def test_validate_string_allowed_chars(self):
    result = validate_string("hello123", "test", allowed_chars=r"^[a-z0-9]+$")
    self.assertEqual(result, "hello123")

  def test_validate_string_disallowed_chars(self):
    with self.assertRaises(ValidationError):
      validate_string("hello!", "test", allowed_chars=r"^[a-z0-9]+$")

  def test_validate_string_sql_injection(self):
    with self.assertRaises(SecurityValidationError):
      validate_string("SELECT * FROM users", "test")

  def test_validate_string_command_injection(self):
    with self.assertRaises(SecurityValidationError):
      validate_string("ls; rm -rf /", "test")

  def test_validate_string_path_traversal(self):
    with self.assertRaises(SecurityValidationError):
      validate_string("../../etc/passwd", "test")

  def test_validate_string_xss(self):
    with self.assertRaises(SecurityValidationError):
      validate_string("<script>alert(1)</script>", "test")

  # -------------------------------------------------------------------------
  # Filename Validation Tests
  # -------------------------------------------------------------------------

  def test_validate_filename_basic(self):
    result = validate_filename("test.txt", "file")
    self.assertEqual(result, "test.txt")

  def test_validate_filename_path_traversal(self):
    with self.assertRaises(SecurityValidationError):
      validate_filename("../test.txt", "file")

  def test_validate_filename_absolute_path(self):
    with self.assertRaises(SecurityValidationError):
      validate_filename("/etc/passwd", "file")

  def test_validate_filename_allowed_extensions(self):
    result = validate_filename("test.txt", "file", allowed_extensions=[".txt", ".csv"])
    self.assertEqual(result, "test.txt")

  def test_validate_filename_disallowed_extension(self):
    with self.assertRaises(ValidationError):
      validate_filename("test.exe", "file", allowed_extensions=[".txt", ".csv"])

  # -------------------------------------------------------------------------
  # Path Validation Tests
  # -------------------------------------------------------------------------

  def test_validate_path_basic(self):
    result = validate_path("test.txt", "path")
    self.assertTrue(os.path.isabs(result))

  def test_validate_path_traversal(self):
    with self.assertRaises(SecurityValidationError):
      validate_path("../test.txt", "path")

  def test_validate_path_allowed_base_dirs(self):
    with self.assertRaises(SecurityValidationError):
      validate_path("/etc/passwd", "path", allowed_base_dirs=["/home/user"])

  # -------------------------------------------------------------------------
  # Numeric Validation Tests
  # -------------------------------------------------------------------------

  def test_validate_int_basic(self):
    result = validate_int("42", "test")
    self.assertEqual(result, 42)

  def test_validate_int_min_max(self):
    result = validate_int(50, "test", min_value=0, max_value=100)
    self.assertEqual(result, 50)

  def test_validate_int_below_min(self):
    with self.assertRaises(ValidationError):
      validate_int(-1, "test", min_value=0)

  def test_validate_int_above_max(self):
    with self.assertRaises(ValidationError):
      validate_int(101, "test", max_value=100)

  def test_validate_int_rejects_bool(self):
    with self.assertRaises(ValidationError):
      validate_int(True, "test")

  def test_validate_float_basic(self):
    result = validate_float("3.14", "test")
    self.assertEqual(result, 3.14)

  def test_validate_float_nan_not_allowed(self):
    with self.assertRaises(ValidationError):
      validate_float(float("nan"), "test")

  def test_validate_float_nan_allowed(self):
    result = validate_float(float("nan"), "test", allow_nan=True)
    self.assertTrue(np.isnan(result))

  def test_validate_float_inf_not_allowed(self):
    with self.assertRaises(ValidationError):
      validate_float(float("inf"), "test")

  def test_validate_float_rejects_bool(self):
    with self.assertRaises(ValidationError):
      validate_float(True, "test")

  # -------------------------------------------------------------------------
  # Tensor/Array Validation Tests
  # -------------------------------------------------------------------------

  def test_validate_shape(self):
    arr = np.zeros((3, 4, 5))
    result = validate_shape(arr, (3, 4, 5), "test")
    self.assertEqual(result, (3, 4, 5))

  def test_validate_shape_none_dimension(self):
    arr = np.zeros((3, 4, 5))
    result = validate_shape(arr, (3, None, 5), "test")
    self.assertEqual(result, (3, 4, 5))

  def test_validate_shape_mismatch(self):
    arr = np.zeros((3, 4, 5))
    with self.assertRaises(ValidationError):
      validate_shape(arr, (3, 5, 5), "test")

  def test_validate_dtype(self):
    arr = np.zeros((3, 4), dtype=np.float32)
    result = validate_dtype(arr, [np.float32, np.float64], "test")
    self.assertEqual(result, np.float32)

  def test_validate_dtype_mismatch(self):
    arr = np.zeros((3, 4), dtype=np.int32)
    with self.assertRaises(ValidationError):
      validate_dtype(arr, [np.float32, np.float64], "test")

  def test_validate_tensor_comprehensive(self):
    arr = np.zeros((2, 3), dtype=np.float32)
    result = validate_tensor(
        arr, "test",
        expected_shape=(2, 3),
        allowed_dtypes=[np.float32],
        min_value=0.0,
        max_value=1.0,
    )
    self.assertIs(result, arr)

  def test_validate_tensor_nan_rejected(self):
    arr = np.array([1.0, np.nan, 3.0])
    with self.assertRaises(ValidationError):
      validate_tensor(arr, "test", allow_nan=False)

  # -------------------------------------------------------------------------
  # Collection Validation Tests
  # -------------------------------------------------------------------------

  def test_validate_list_basic(self):
    result = validate_list([1, 2, 3], "test")
    self.assertEqual(result, [1, 2, 3])

  def test_validate_list_element_validator(self):
    result = validate_list(
        ["1", "2", "3"], "test",
        element_validator=lambda x: validate_int(x, min_value=0)
    )
    self.assertEqual(result, [1, 2, 3])

  def test_validate_list_unique(self):
    with self.assertRaises(ValidationError):
      validate_list([1, 2, 2], "test", unique=True)

  def test_validate_dict_basic(self):
    result = validate_dict({"a": 1, "b": 2}, "test")
    self.assertEqual(result, {"a": 1, "b": 2})

  def test_validate_dict_required_keys(self):
    with self.assertRaises(ValidationError):
      validate_dict({"a": 1}, "test", required_keys=["a", "b"])

  def test_validate_dict_allowed_keys(self):
    with self.assertRaises(ValidationError):
      validate_dict({"a": 1, "c": 3}, "test", allowed_keys=["a", "b"])

  def test_validate_dict_key_value_validators(self):
    result = validate_dict(
        {"name": "test", "value": "42"},
        "test",
        key_validator=lambda k: validate_string(k, allowed_chars=r"^[a-z]+$"),
        value_validator=lambda v: validate_int(v, min_value=0)
    )
    self.assertEqual(result, {"name": "test", "value": 42})

  # -------------------------------------------------------------------------
  # Config Validation Tests
  # -------------------------------------------------------------------------

  def test_validate_config(self):
    schema = {
        "name": {"validator": lambda x: validate_string(x, max_length=50)},
        "count": {"validator": lambda x: validate_int(x, min_value=0), "required": True},
        "enabled": {"validator": lambda x: bool(x), "default": True},
    }
    config = {"name": "test", "count": 5}
    result = validate_config(config, schema, "test")
    self.assertEqual(result["name"], "test")
    self.assertEqual(result["count"], 5)
    self.assertEqual(result["enabled"], True)

  def test_validate_config_missing_required(self):
    schema = {"required_field": {"required": True}}
    with self.assertRaises(ValidationError):
      validate_config({}, schema, "test")

  def test_validate_config_unexpected_key(self):
    schema = {"allowed": {}}
    with self.assertRaises(ValidationError):
      validate_config({"allowed": 1, "unexpected": 2}, schema, "test")

  # -------------------------------------------------------------------------
  # Sanitization Tests
  # -------------------------------------------------------------------------

  def test_sanitize_string_control_chars(self):
    result = sanitize_string("hello\x00world\x01")
    self.assertEqual(result, "helloworld")

  def test_sanitize_string_truncation(self):
    result = sanitize_string("x" * 20000, max_length=100)
    self.assertEqual(len(result), 100)

  def test_sanitize_path(self):
    result = sanitize_path("subdir/file.txt", "/home/user")
    self.assertEqual(result, "/home/user/subdir/file.txt")

  def test_sanitize_path_traversal(self):
    with self.assertRaises(SecurityValidationError):
      sanitize_path("../../etc/passwd", "/home/user")

  # -------------------------------------------------------------------------
  # Decorator Tests
  # -------------------------------------------------------------------------

  def test_validate_args_decorator(self):
    @validate_args(
        x=lambda v: validate_int(v, min_value=0, max_value=100),
        name=lambda v: validate_string(v, max_length=20),
    )
    def func(x, name):
      return x, name

    result = func(50, "test")
    self.assertEqual(result, (50, "test"))

  def test_validate_args_decorator_failure(self):
    @validate_args(
        x=lambda v: validate_int(v, min_value=0, max_value=100),
    )
    def func(x):
      return x

    with self.assertRaises(ValidationError):
      func(150)


if __name__ == "__main__":
  googletest.main()