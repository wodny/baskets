#!/usr/bin/env python3

# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>. 

# Author: Marcin Szewczyk <marcin.szewczyk@wodny.org>

import sys
import re
import datetime
import argparse
import dataclasses
import operator
import itertools
import functools
import string
from collections import defaultdict
from collections.abc import Callable
from typing import cast, Generator, Iterable, Optional, Any, Union
import logging

__version__ = "0.1"

logger = logging.getLogger(__name__)

ConvertedValue = Union[str, int, datetime.datetime]

@dataclasses.dataclass
class Filter:
    arg1: Optional[str]
    op: Callable[..., bool]
    arg2: Optional[ConvertedValue]

    known_ops = {
        "<=": operator.le,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne,
        "=":  operator.eq,
        "<":  operator.lt,
        ">":  operator.gt,
    } # type: dict[str, Callable[..., bool]]

    @classmethod
    def from_str(cls, rule: Optional[str], converter: "Converter") -> "Filter":
        # TODO logic operators and parentheses
        if not rule:
            return cls(None, lambda a, b: True, None)
        for known_op in Filter.known_ops:
            arg1, op, arg2 = rule.partition(known_op)
            if arg2:
                return cls(arg1, Filter.known_ops[known_op], converter(arg1, arg2))
        raise ValueError(f"operator not found in {rule}")

    def __contains__(self, pattern_match: Optional["PatternMatch"]) -> bool:
        if pattern_match is None:
            return False
        return self.op(pattern_match[self.arg1] if self.arg1 else None, self.arg2)

@dataclasses.dataclass
class BasketPattern:
    template: string.Template

    @staticmethod
    @functools.lru_cache
    def get_datetime_dict(name: str, dt: datetime.datetime) -> dict[str, str]:
        # TODO support everything (?) from strftime
        dt_dict = {
            "Y": f"{dt.year:04d}",
            "m": f"{dt.month:02d}",
            "d": f"{dt.day:02d}",
        }
        return {
            f"{name}__{symbol}": value
            for symbol, value
            in dt_dict.items()
        }

    @classmethod
    def from_str(cls, pattern: Optional[str]) -> "BasketPattern":
        return cls(string.Template(pattern or ""))

    def substitute(self, mapping: dict[str, ConvertedValue]) -> str:
        mapping_copy = dict(mapping)
        for name, value in mapping.items():
            if isinstance(value, datetime.datetime):
                mapping_copy.update(BasketPattern.get_datetime_dict(name, value))
        return self.template.substitute(mapping_copy)

@dataclasses.dataclass
class Selection:
    rule: Optional[str]
    fn: Callable[[Iterable["PatternMatch"]], Iterable["PatternMatch"]]

    @classmethod
    def from_str(cls, rule: Optional[str], invert: bool) -> "Selection":
        # TODO more filters
        if not rule:
            # all/none filter
            if invert:
                def filter_(iterable: Iterable["PatternMatch"]) -> Iterable["PatternMatch"]:
                    return []
            else:
                def filter_(iterable: Iterable["PatternMatch"]) -> Iterable["PatternMatch"]:
                    return iterable
        elif rule[0] == "%":
            # modulo filter
            mod = int(rule[1:])
            def filter_(iterable: Iterable["PatternMatch"]) -> Iterable["PatternMatch"]:
                for i, elem in enumerate(iterable):
                    if (i % mod == 0) ^ invert:
                        yield elem
        else:
            # head filter
            count = int(rule)
            def filter_(iterable: Iterable["PatternMatch"]) -> Iterable["PatternMatch"]:
                if invert:
                    return itertools.islice(iterable, count, None)
                else:
                    return itertools.islice(iterable, count)

        return cls(rule, filter_)
    
@dataclasses.dataclass
class Basket:
    name: str
    samples: list["PatternMatch"] = dataclasses.field(init=False, default_factory=list)

    def add(self, sample: "PatternMatch") -> None:
        self.samples.append(sample)

    def sort(self) -> None:
        self.samples.sort(key=lambda pm: pm.text)

@dataclasses.dataclass
class BasketGroup:
    name: str
    filter_: Filter
    pattern: BasketPattern
    selection: Selection
    baskets: dict[str, Basket] = dataclasses.field(init=False, default_factory=dict)

    @classmethod
    def from_str(cls, name: str, rule: str, converter: "Converter", invert: bool) -> "BasketGroup":
        tokens = [] # type: list[Optional[str]]
        tokens.extend(rule.split(":", maxsplit=2))
        tokens.extend([None] * (3 - len(tokens)))
        filter_, pattern, selection = tokens
        return cls(
            name,
            Filter.from_str(filter_, converter),
            BasketPattern.from_str(pattern),
            Selection.from_str(selection, invert)
        )

    def __contains__(self, pattern_match: "PatternMatch") -> bool:
        return pattern_match in self.filter_

    def get_basket_name(self, pattern_match: "PatternMatch") -> str:
        return self.pattern.substitute(pattern_match.groups)

    def add_sample(self, pattern_match: "PatternMatch") -> None:
        basket_name = self.get_basket_name(pattern_match)
        if basket_name not in self.baskets:
            self.baskets[basket_name] = Basket(basket_name)
        self.baskets[basket_name].add(pattern_match)

    def sort(self) -> None:
        for basket in self.baskets.values():
            basket.sort()

    def select(self, basket: Basket) -> Iterable["PatternMatch"]:
        return self.selection.fn(basket.samples)

@dataclasses.dataclass
class Pattern:
    regex: re.Pattern[str]
    types: dict[str, Optional[str]]

    @staticmethod
    def get_name_type(value: str) -> tuple[str, str]:
        name, _, type_ = value.partition("__")
        return name, type_

    @classmethod
    def from_str(cls, rule: str) -> "Pattern":
        regex = re.compile(rule)
        types = {
            name: type_ or None
            for name, type_
            in map(Pattern.get_name_type, regex.groupindex)
        }
        return cls(regex, types)

    def match(self, text: str, converter: "Converter") -> Optional["PatternMatch"]:
        if m := self.regex.match(text):
            def iter_name_value() -> Generator[tuple[str, ConvertedValue], None, None]:
                for fullname, value in cast(re.Match[str], m).groupdict().items():
                    name, _ = Pattern.get_name_type(fullname)
                    value = converter(name, value)
                    yield name, value
            groups = dict(iter_name_value())
            return PatternMatch(text, self, groups)

@dataclasses.dataclass
class PatternMatch:
    text: str
    regex: Pattern
    groups: dict[str, ConvertedValue]

    def __getitem__(self, name: str) -> ConvertedValue:
        return self.groups[name]

@dataclasses.dataclass
class Converter:
    pattern: Pattern
    date_format: str

    converters_mapping = {
        "int": lambda v, *args, **kwargs: int(v),
        "dt":  lambda v, date_format, *args, **kwargs: datetime.datetime.strptime(v, date_format),
    } # type: dict[str, Callable[..., ConvertedValue]]

    @classmethod
    def from_pattern(cls, pattern: Pattern, date_format: str) -> "Converter":
        return cls(pattern, date_format)

    def __call__(self, name: str, value: str) -> ConvertedValue:
        type_ = self.pattern.types[name]
        converter = None if type_ is None else self.converters_mapping.get(type_)
        return value if converter is None else converter(value, date_format=self.date_format)

@dataclasses.dataclass
class Output:
    last_count: Optional[int] = dataclasses.field(init=False, default=None)

    def output_basket_group(self, basket_group: BasketGroup) -> Optional[str]:
        pass

    def output_basket(self, basket: Basket) -> Optional[str]:
        pass
    
    def output_sample(self, sample: PatternMatch) -> Optional[str]:
        pass

    def __call__(self, basket_groups: list[BasketGroup]) -> Generator[Optional[str], None, None]:
        self.last_count = 0
        for i, basket_group in enumerate(basket_groups):
            yield self.output_basket_group(basket_group)
            for basket in basket_group.baskets.values():
                yield self.output_basket(basket)
                for sample in basket_group.select(basket):
                    self.last_count += 1
                    yield self.output_sample(sample)

@dataclasses.dataclass
class OutputHuman(Output):
    def output_basket_group(self, basket_group: BasketGroup) -> str:
        return f"Basket group: {basket_group.name}"

    def output_basket(self, basket: Basket) -> str:
        return f"  Basket: {basket.name}"
    
    def output_sample(self, sample: PatternMatch) -> str:
        return f"    Sample: {sample.text}"

# TODO add nul-delimited output
@dataclasses.dataclass
class OutputLines(Output):
    def output_sample(self, sample: PatternMatch) -> str:
        return sample.text

def get_options() -> argparse.Namespace:
    output_types = {
        "human": OutputHuman,
        "lines": OutputLines,
    }
    
    p = argparse.ArgumentParser()
    p.add_argument("--version", action="version", version="%(prog)s " + __version__)
    p.add_argument("-l", "--loglevel", default="INFO", type=str.upper, help="log level")
    p.add_argument("-i", "--invert", action="store_true", help="invert selection")
    p.add_argument("-d", "--date-format", default="%Y-%m-%d", help="input date format")
    p.add_argument("-b", "--basket", dest="baskets", action="append", default=[], help="basket group specification")
    p.add_argument(
        "-o", "--output",
        type=lambda v: output_types[v](),
        default="human",
        help="output mode ({})".format(", ".join(output_types))
    )
    p.add_argument("pattern", type=Pattern.from_str, help="input lines (sample names) regex pattern")
    options = p.parse_args()
    return options

def main() -> None:
    options = get_options()
    
    logging.basicConfig(level=options.loglevel)
    converter = Converter(options.pattern, options.date_format)
    basket_groups = [
        BasketGroup.from_str(f"{i}", rule, converter, options.invert)
        for i, rule
        in enumerate(options.baskets)
    ]

    lines_matched, lines_not_matched, lines_in_baskets = 0, 0, 0
    for line in map(lambda v: v.rstrip(), sys.stdin):
        if m := options.pattern.match(line, converter):
            for basket_group in basket_groups:
                if m in basket_group:
                    basket_group.add_sample(m)
                    lines_in_baskets += 1
                    break
            else:
                logger.debug("no basket found for line: %s", line)
            lines_matched += 1
        else:
            logger.debug("line not matched: %s", line)
            lines_not_matched += 1

    for basket_group in basket_groups:
        basket_group.sort()

    for line in filter(None, options.output(basket_groups)):
        print(line)

    lines_all = lines_matched + lines_not_matched
    logger.info("lines matching pattern: %d of %d", lines_matched, lines_all)
    logger.info("lines matching basket groups conditions: %d of %d", lines_in_baskets, lines_all)
    logger.info("samples in output: %d", options.output.last_count)

if __name__ == "__main__":
    main()
