"""No-driver parser fixtures: synthetic metadata is never CUDA execution proof."""

import hashlib
import struct
import unittest
from unittest import mock

from xvram import compat_audit_binary as binary


_ELF_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
_SECTION = struct.Struct("<IIQQQQIIQQ")
_SYMBOL = struct.Struct("<IBBHQQ")
_U64_MAX = (1 << 64) - 1


class _ElfFixture:
    """Tiny ELF64LE executable with ordinary CUDA-named symbol sections."""

    def __init__(self, name="foo", *, duplicate_symbol=False, include_info=True):
        self.section_names = ["", ".shstrtab", ".strtab", ".symtab", ".text." + name]
        if include_info:
            self.section_names.append(".nv.info." + name)
        names = bytearray()
        name_offsets = []
        for section_name in self.section_names:
            name_offsets.append(len(names))
            names.extend(section_name.encode("ascii") + b"\0")
        strings = b"\0" + name.encode("ascii") + b"\0"
        symbol = _SYMBOL.pack(1, 0x12, 0, 4, 0, 16)
        symbols = bytes(24) + symbol + (symbol if duplicate_symbol else b"")
        payloads = [b"", bytes(names), strings, symbols, bytes(range(16))]
        if include_info:
            payloads.append(b"\x04\x00")
        section_count = len(payloads)
        self.data = bytearray(64 + section_count * 64)
        ident = b"\x7fELF\x02\x01\x01" + bytes(9)
        _ELF_HEADER.pack_into(self.data, 0, ident, 2, 190, 1, 0, 0, 64, 0x55,
                              64, 0, 0, 64, section_count, 1)
        for index, payload in enumerate(payloads):
            if index == 0:
                fields = (0,) * 10
            else:
                offset = len(self.data)
                self.data.extend(payload)
                section_type = 3 if index in (1, 2) else 2 if index == 3 else 1
                flags = 6 if index == 4 else 0
                fields = (name_offsets[index], section_type, flags, 0, offset,
                          len(payload), 2 if index == 3 else 0, 1 if index == 3 else 0,
                          8 if index in (3, 4) else 1, 24 if index == 3 else 0)
            _SECTION.pack_into(self.data, 64 + index * 64, *fields)

    def section(self, index):
        return list(_SECTION.unpack_from(self.data, 64 + index * 64))

    def set_section(self, index, field, value):
        values = self.section(index)
        values[field] = value
        _SECTION.pack_into(self.data, 64 + index * 64, *values)
        return self

    def set_symbol(self, field, value, index=1):
        offset = self.section(3)[4] + index * 24
        values = list(_SYMBOL.unpack_from(self.data, offset))
        values[field] = value
        _SYMBOL.pack_into(self.data, offset, *values)
        return self

    def header(self, offset, encoding, value):
        struct.pack_into(encoding, self.data, offset, value)
        return self

    def bytes(self):
        return bytes(self.data)


def _parameter(ordinal, offset, size, *, index=0):
    return ["Attribute: EIATTR_KPARAM_INFO",
            f"Value: Index : 0x{index:x} Ordinal : 0x{ordinal:x} "
            f"Offset : 0x{offset:x} Size : 0x{size:x}"]


def _section(name="foo", bank=16, parameters=((0, 0, 8), (1, 8, 8))):
    lines = [".nv.info." + name, "Attribute: EIATTR_CBANK_PARAM_SIZE",
             f"Value: 0x{bank:x}"]
    for ordinal, offset, size in parameters:
        lines.extend(_parameter(ordinal, offset, size))
    return lines


def _parse(lines, names=("foo",)):
    parser = binary.ParameterDump(names)
    for line in lines:
        parser.line(line if isinstance(line, bytes) else (line + "\n").encode("ascii"))
    return parser.finish()


class ElfFunctionTests(unittest.TestCase):
    def assert_rejected(self, fixture, code=None):
        data = fixture.bytes() if isinstance(fixture, _ElfFixture) else fixture
        if code is None:
            with self.assertRaises(binary.BinaryAuditError):
                binary.elf_functions(data)
        else:
            with self.assertRaisesRegex(binary.BinaryAuditError, "^" + code + "$"):
                binary.elf_functions(data)

    def test_cuda_elf_symbols_are_offline_metadata_only(self):
        data = _ElfFixture().bytes()
        result = binary.elf_functions(data)
        self.assertEqual(result, {"sha256": hashlib.sha256(data).hexdigest(),
                                 "size_bytes": len(data), "elf_flags": 0x55,
                                 "functions": {"foo": {"symbol_index": 1,
                                     "section_index": 4, "code_bytes": 16}}})

    def test_nv_info_not_required_for_symbol_inventory(self):
        result = binary.elf_functions(_ElfFixture(include_info=False).bytes())
        self.assertEqual(set(result["functions"]), {"foo"})

    def test_non_function_and_undefined_symbols_are_not_functions(self):
        for field, value in ((1, 0x11), (3, 0)):
            with self.subTest(field=field):
                self.assertEqual(binary.elf_functions(
                    _ElfFixture().set_symbol(field, value).bytes())["functions"], {})

    def test_different_named_function_section_is_not_a_match(self):
        fixture = _ElfFixture().set_symbol(3, 5)
        self.assertEqual(binary.elf_functions(fixture.bytes())["functions"], {})

    def test_input_type_and_size_limits(self):
        for data in (None, "ELF", bytearray(64), b"", bytes(63)):
            with self.subTest(type=type(data).__name__):
                self.assert_rejected(data, "cubin_size_limit")
        data = _ElfFixture().bytes()
        with mock.patch.object(binary, "MAX_CUBIN_BYTES", len(data) - 1):
            self.assert_rejected(data, "cubin_size_limit")

    def test_unsupported_magic_class_endianness_and_ident_version(self):
        for offset, value in ((0, 0), (4, 1), (5, 2), (6, 0)):
            with self.subTest(offset=offset):
                fixture = _ElfFixture()
                fixture.data[offset] = value
                self.assert_rejected(fixture, "unsupported_elf_encoding")

    def test_wrong_cuda_machine_kind_header_size_and_version(self):
        for offset, encoding, value in ((16, "<H", 1), (18, "<H", 62),
                                         (20, "<I", 2), (52, "<H", 63)):
            with self.subTest(offset=offset):
                self.assert_rejected(_ElfFixture().header(offset, encoding, value),
                                     "unsupported_cuda_elf_header")

    def test_section_table_forms_are_not_guessed(self):
        for offset, value in ((58, 63), (60, 0), (62, 0), (62, 6), (62, 0xffff)):
            with self.subTest(offset=offset, value=value):
                self.assert_rejected(_ElfFixture().header(offset, "<H", value),
                                     "unsupported_elf_section_table")

    def test_truncated_headers_tables_and_payloads(self):
        data = _ElfFixture().bytes()
        for length in (0, 15, 63, 64, 128, 64 + 6 * 64 - 1, len(data) - 1):
            with self.subTest(length=length):
                self.assert_rejected(data[:length])

    def test_section_offset_size_and_table_multiplication_bounds(self):
        fixtures = [_ElfFixture().header(40, "<Q", _U64_MAX),
                    _ElfFixture().header(60, "<H", 65535),
                    _ElfFixture().set_section(4, 4, _U64_MAX),
                    _ElfFixture().set_section(4, 5, _U64_MAX)]
        for fixture in fixtures:
            with self.subTest(data=fixture.section(4)[4:6]):
                self.assert_rejected(fixture, "elf_span_out_of_bounds")

    def test_section_name_table_type_and_offsets(self):
        self.assert_rejected(_ElfFixture().set_section(1, 1, 1),
                             "elf_section_names_not_strings")
        fixture = _ElfFixture()
        self.assert_rejected(fixture.set_section(4, 0, fixture.section(1)[5]),
                             "elf_string_offset_out_of_bounds")

    def test_unterminated_non_ascii_and_oversized_section_names(self):
        fixture = _ElfFixture()
        strings = fixture.section(1)
        fixture.data[strings[4] + strings[5] - 1] = ord("x")
        self.assert_rejected(fixture, "elf_string_unterminated_or_too_long")
        fixture = _ElfFixture()
        fixture.data[fixture.section(1)[4] + 1] = 255
        self.assert_rejected(fixture, "elf_string_not_ascii")
        self.assert_rejected(_ElfFixture("f" * 4097),
                             "elf_string_unterminated_or_too_long")

    def test_duplicate_section_names(self):
        fixture = _ElfFixture()
        fixture.set_section(5, 0, fixture.section(4)[0])
        self.assert_rejected(fixture, "duplicate_elf_section_name")

    def test_symbol_table_must_be_unique_and_well_sized(self):
        for fixture in (_ElfFixture().set_section(3, 1, 1),
                        _ElfFixture().set_section(5, 1, 2)):
            self.assert_rejected(fixture, "elf_symbol_table_not_unique")
        for field, value in ((9, 16), (5, 47)):
            self.assert_rejected(_ElfFixture().set_section(3, field, value),
                                 "invalid_elf_symbol_table")
        with mock.patch.object(binary, "MAX_SYMBOLS", 1):
            self.assert_rejected(_ElfFixture(), "invalid_elf_symbol_table")

    def test_symbol_string_table_link_is_checked(self):
        for link in (0, 6, 4):
            with self.subTest(link=link):
                self.assert_rejected(_ElfFixture().set_section(3, 6, link),
                                     "invalid_elf_symbol_strings")

    def test_symbol_name_offset_termination_and_ascii(self):
        fixture = _ElfFixture()
        self.assert_rejected(fixture.set_symbol(0, fixture.section(2)[5]),
                             "elf_string_offset_out_of_bounds")
        fixture = _ElfFixture()
        strings = fixture.section(2)
        fixture.data[strings[4] + strings[5] - 1] = ord("x")
        self.assert_rejected(fixture, "elf_string_unterminated_or_too_long")
        fixture = _ElfFixture()
        fixture.data[fixture.section(2)[4] + 1] = 255
        self.assert_rejected(fixture, "elf_string_not_ascii")

    def test_duplicate_or_private_function_names_are_rejected(self):
        self.assert_rejected(_ElfFixture(duplicate_symbol=True), "duplicate_elf_function")
        for name in ("0x12345678", "0XABCDEF12", "foo/bad", "f" * 1025):
            with self.subTest(name=name[:30]):
                self.assert_rejected(_ElfFixture(name), "invalid_elf_function")

    def test_function_index_size_and_address_span_are_checked(self):
        for index in (6, 0xffff):
            self.assert_rejected(_ElfFixture().set_symbol(3, index),
                                 "unsupported_function_section_index")
        self.assert_rejected(_ElfFixture().set_symbol(5, 17))
        for value, size in ((1, 16), (17, 1), (_U64_MAX, 1)):
            with self.subTest(value=value, size=size):
                self.assert_rejected(_ElfFixture().set_symbol(4, value).set_symbol(5, size))
        self.assert_rejected(_ElfFixture().set_section(4, 3, 32).set_symbol(4, 31))

    def test_function_section_must_have_executable_file_backing(self):
        for section_type, flags in ((8, 6), (3, 6), (1, 0), (1, 2)):
            with self.subTest(section_type=section_type, flags=flags):
                self.assert_rejected(_ElfFixture().set_section(4, 1, section_type)
                                     .set_section(4, 2, flags))

    def test_function_address_is_relative_to_its_section_address(self):
        fixture = _ElfFixture().set_section(4, 3, 32).set_symbol(4, 32)
        self.assertEqual(binary.elf_functions(fixture.bytes())["functions"]["foo"]["code_bytes"], 16)


class ParameterDumpTests(unittest.TestCase):
    def assert_rejected(self, lines, code, names=("foo",)):
        with self.assertRaisesRegex(binary.BinaryAuditError, "^" + code + "$"):
            _parse(lines, names)

    def test_exact_parameter_slots_without_pointee_inference(self):
        lines = _section()
        lines[5:5] = ["Pointee's logAlignment : 0x3", "Space : Global", "ReadOnly : 0x1"]
        self.assertEqual(_parse(lines), {"foo": {"parameter_bytes": 16,
            "parameters": [{"ordinal": 0, "offset_bytes": 0, "size_bytes": 8},
                           {"ordinal": 1, "offset_bytes": 8, "size_bytes": 8}]}})

    def test_selected_names_require_a_bounded_explicit_collection(self):
        for names in ("foo", None, iter(("foo",)),
                      ["f" + str(index) for index in range(4097)]):
            with self.subTest(kind=type(names).__name__):
                with self.assertRaisesRegex(binary.BinaryAuditError, "^selected_name_limit$"):
                    binary.ParameterDump(names)
        for names in (["foo"], ("foo",), {"foo"}, frozenset(("foo",))):
            with self.subTest(kind=type(names).__name__):
                self.assertEqual(set(_parse(_section(), names)), {"foo"})
        for names in ([], (), set(), frozenset()):
            with self.subTest(empty_kind=type(names).__name__):
                self.assertEqual(_parse(_section(), names), {})
        names = tuple("f" + str(index) for index in range(4096))
        self.assertEqual(len(binary.ParameterDump(names).names), 4096)

    def test_selected_names_are_unique_valid_and_address_free(self):
        for names in (["foo", "foo"], ("foo", "foo")):
            with self.assertRaisesRegex(binary.BinaryAuditError, "^duplicate_selected_name$"):
                binary.ParameterDump(names)
        for name in (None, 1, True, b"foo", "", "foo\0", "foo\n", "foo/bad",
                     "foo\\bad", "f" * 1025, "0x12345678", "foo0XABCDEF", "f\u00e9"):
            with self.subTest(name=repr(name)):
                with self.assertRaisesRegex(binary.BinaryAuditError, "^invalid_selected_name$"):
                    binary.ParameterDump([name])

    def test_alignment_padding_is_not_a_missing_ordinal(self):
        layout = _parse(_section(parameters=((1, 8, 8), (0, 0, 4))))["foo"]
        self.assertEqual([item["ordinal"] for item in layout["parameters"]], [0, 1])
        self.assertEqual(layout["parameters"][1]["offset_bytes"], 8)

    def test_streaming_crlf_indentation_and_multiple_selected_sections(self):
        parser = binary.ParameterDump(("foo", "bar"))
        for line in _section() + _section("bar", 4, ((0, 0, 4),)) + [".text.bar"]:
            parser.line(("\t" + line + "\r\n").encode("ascii"))
        self.assertEqual(set(parser.finish()), {"foo", "bar"})
        self.assertEqual(parser.finish()["bar"]["parameter_bytes"], 4)

    def test_unselected_sections_and_unknown_attributes_do_not_supply_layouts(self):
        lines = [".nv.info.other", "Attribute: EIATTR_KPARAM_INFO", "Value: malformed",
                 ".text.other", "Value: 0xffffffff"]
        lines += _section()
        lines += ["Attribute: EIATTR_UNKNOWN", "Value: opaque ignored metadata",
                  ".nv.info.other", "Attribute: EIATTR_CBANK_PARAM_SIZE", "Value: invalid"]
        self.assertEqual(set(_parse(lines)), {"foo"})

    def test_missing_selected_section_and_layout_parts(self):
        self.assert_rejected(_section("bar"), "selected_parameter_section_missing")
        self.assert_rejected(_section(), "selected_parameter_section_missing", ("foo", "bar"))
        self.assert_rejected([".nv.info.foo"] + _parameter(0, 0, 8), "missing_parameter_layout")
        self.assert_rejected(_section(parameters=()), "missing_parameter_layout")

    def test_duplicate_sections_bank_and_ordinals(self):
        self.assert_rejected(_section() + _section(), "duplicate_parameter_section")
        self.assert_rejected(_section() + ["Attribute: EIATTR_CBANK_PARAM_SIZE", "Value: 0x10"],
                             "invalid_parameter_bank_size")
        self.assert_rejected(_section() + _parameter(1, 8, 8), "duplicate_parameter_ordinal")

    def test_ordinal_gaps_overlaps_and_bank_extents_are_not_guessed(self):
        for bank, parameters, code in (
            (16, ((1, 0, 16),), "noncontiguous_parameter_ordinals"),
            (16, ((0, 0, 8), (2, 8, 8)), "noncontiguous_parameter_ordinals"),
            (8, ((0, 0, 8), (1, 4, 4)), "overlapping_parameter_ranges"),
            (8, ((0, 0, 16),), "parameter_exceeds_bank"),
            (16, ((0, 0, 8),), "parameter_bank_extent_mismatch"),
        ):
            with self.subTest(code=code):
                self.assert_rejected(_section(bank=bank, parameters=parameters), code)

    def test_truncated_attribute_rejected_at_eof_attribute_and_section_boundaries(self):
        for attribute in ("EIATTR_KPARAM_INFO", "EIATTR_CBANK_PARAM_SIZE"):
            for suffix in ([], ["Attribute: EIATTR_UNKNOWN"], [".nv.info.other"], [".text.foo"]):
                with self.subTest(attribute=attribute, suffix=suffix):
                    self.assert_rejected(_section() + ["Attribute: " + attribute] + suffix,
                                         "truncated_parameter_attribute")

    def test_bank_size_encoding_and_limits(self):
        for value in ("16", "0X10", "0x100000000", "0x10 extra", "", "0x-1"):
            with self.subTest(value=value):
                self.assert_rejected([".nv.info.foo", "Attribute: EIATTR_CBANK_PARAM_SIZE", "Value: " + value],
                                     "invalid_parameter_bank_size")
        for value in (0, binary.MAX_PARAMETER_BYTES + 1, 0xffffffff):
            self.assert_rejected(_section(bank=value), "parameter_bank_size_limit")

    def test_parameter_metadata_requires_exact_four_hex_fields(self):
        values = ["Index : 0x0 Ordinal : 0x0 Offset : 0x0",
                  "Ordinal : 0x0 Index : 0x0 Offset : 0x0 Size : 0x8",
                  "Index : 0 Ordinal : 0x0 Offset : 0x0 Size : 0x8",
                  "Index : 0x0 Ordinal : 0x0 Offset : 0x0 Size : 0x8 extra",
                  "Index : 0x0 Ordinal : 0x0 Offset : 0x0 Size : 0x100000000"]
        for value in values:
            with self.subTest(value=value):
                self.assert_rejected([".nv.info.foo", "Attribute: EIATTR_KPARAM_INFO", "Value: " + value],
                                     "invalid_parameter_attribute")

    def test_parameter_index_ordinal_size_and_addition_limits(self):
        cases = [(1, 0, 0, 8), (0, binary.MAX_PARAMETERS, 0, 8), (0, 0, 0, 0),
                 (0, 0, 0, binary.MAX_PARAMETER_BYTES + 1),
                 (0, 0, binary.MAX_PARAMETER_BYTES, 1), (0, 0, 0xffffffff, 1)]
        for index, ordinal, offset, size in cases:
            with self.subTest(case=(index, ordinal, offset, size)):
                self.assert_rejected([".nv.info.foo"] + _parameter(ordinal, offset, size, index=index),
                                     "parameter_attribute_limit")

    def test_maximum_parameter_count_and_bank_size_are_bounded(self):
        layout = _parse(_section(bank=binary.MAX_PARAMETERS,
            parameters=tuple((index, index, 1) for index in range(binary.MAX_PARAMETERS))))["foo"]
        self.assertEqual(len(layout["parameters"]), binary.MAX_PARAMETERS)
        layout = _parse(_section(bank=binary.MAX_PARAMETER_BYTES,
            parameters=((0, 0, binary.MAX_PARAMETER_BYTES),)))["foo"]
        self.assertEqual(layout["parameter_bytes"], binary.MAX_PARAMETER_BYTES)

    def test_non_ascii_and_line_limit_apply_even_to_unselected_output(self):
        for prefix in ([], [".nv.info.other"], [".nv.info.foo"]):
            with self.subTest(prefix=prefix):
                self.assert_rejected(prefix + [b"\xff\n"], "non_ascii_tool_output")
                self.assert_rejected(prefix + [b"x" * (binary.MAX_DUMP_LINE + 1)], "dump_line_limit")
        self.assertEqual(set(_parse([b"x" * binary.MAX_DUMP_LINE] + _section())), {"foo"})


class ElfListTests(unittest.TestCase):
    def assert_rejected(self, raw, code):
        with self.assertRaisesRegex(binary.BinaryAuditError, "^" + code + "$"):
            binary.parse_elf_list(raw)

    def test_only_exact_sm86_entries_are_selected(self):
        raw = (b"ELF file 1: ggml-cuda.1.sm_80.cubin\n"
               b"ELF file 2: ggml-cuda.2.sm_86.cubin\n"
               b"ELF file 3: ggml-cuda.3.sm_90a.cubin\n"
               b"ELF file 4: ggml-cuda.4.sm_100f.cubin\n"
               b"ELF file 5: ggml-cuda.5.sm_86.cubin\n")
        self.assertEqual(binary.parse_elf_list(raw),
                         [(2, "ggml-cuda.2.sm_86.cubin"), (5, "ggml-cuda.5.sm_86.cubin")])

    def test_line_endings_whitespace_and_noncontiguous_indices(self):
        raw = (b"\r\n  ELF file\t7: ggml-cuda.7.sm_86.cubin  \r\n\n"
               b"ELF file 999999: ggml-cuda.999999.sm_86.cubin")
        self.assertEqual(binary.parse_elf_list(raw),
                         [(7, "ggml-cuda.7.sm_86.cubin"),
                          (999999, "ggml-cuda.999999.sm_86.cubin")])

    def test_empty_or_other_architecture_is_not_sm86_evidence(self):
        for raw in (b"", b" \n\r\n", b"ELF file 1: ggml-cuda.1.sm_80.cubin\n",
                    b"ELF file 1: ggml-cuda.1.sm_86a.cubin\n"):
            with self.subTest(raw=raw):
                self.assert_rejected(raw, "sm86_cubin_count_limit")

    def test_duplicate_entries_or_indices_in_any_architecture_are_rejected(self):
        entry = b"ELF file 1: ggml-cuda.1.sm_86.cubin\n"
        self.assert_rejected(entry + entry, "duplicate_elf_list_entry")
        self.assert_rejected(entry + b"ELF file 1: ggml-cuda.1.sm_80.cubin\n",
                             "duplicate_elf_list_entry")
        other = b"ELF file 2: ggml-cuda.2.sm_80.cubin\n"
        self.assert_rejected(entry + other + other, "duplicate_elf_list_entry")

    def test_selected_cubin_count_limit_is_enforced(self):
        raw = b"ELF file 1: ggml-cuda.1.sm_86.cubin\nELF file 2: ggml-cuda.2.sm_86.cubin\n"
        with mock.patch.object(binary, "MAX_CUBINS", 1):
            self.assert_rejected(raw, "sm86_cubin_count_limit")

    def test_malformed_or_warning_lines_are_not_skipped(self):
        values = [b"warning: skipped image", b"ELF file 1: ggml-cuda.2.sm_86.cubin",
                  b"ELF file 01: ggml-cuda.01.sm_86.cubin", b"ELF file 0: ggml-cuda.0.sm_86.cubin",
                  b"ELF file 1000000: ggml-cuda.1000000.sm_86.cubin",
                  b"ELF file 1: ggml-cuda.1.sm_86.cubin extra",
                  b"ELF file 1: ggml-cuda.1.sm_86.cubin\0",
                  b"ELF file 1: other.1.sm_86.cubin", b"ELF file 1: ggml-cuda.1.sm_8.cubin"]
        for value in values:
            with self.subTest(value=value):
                self.assert_rejected(value, "invalid_elf_list_entry")

    def test_paths_cannot_escape_fresh_extraction_directory(self):
        names = [b"../ggml-cuda.1.sm_86.cubin", b"sub/ggml-cuda.1.sm_86.cubin",
                 b"sub\\ggml-cuda.1.sm_86.cubin", b"/ggml-cuda.1.sm_86.cubin",
                 b"C:\\ggml-cuda.1.sm_86.cubin", b"ggml-cuda.1.sm_86.cubin:stream"]
        for name in names:
            with self.subTest(name=name):
                self.assert_rejected(b"ELF file 1: " + name, "invalid_elf_list_entry")

    def test_encoding_and_total_output_limit(self):
        self.assert_rejected(b"\xff\n", "elf_list_encoding")
        self.assert_rejected(b" " * (1024 * 1024 + 1), "elf_list_limit")


if __name__ == "__main__":
    unittest.main()
