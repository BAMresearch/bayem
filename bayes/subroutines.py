def len_or_one(obj):
    """Returns the length of an object or 1 if no length is defined."""
    if hasattr(obj, '__len__'):
        length = len(obj)
    else:
        length = 1
    return length



def underlined_string(string, symbol="=", n_empty_start=1, n_empty_end=1):
    n_chars = len(string)
    underline_string = n_chars * symbol
    empty_lines_start = n_empty_start * "\n"
    empty_lines_end = n_empty_end * "\n"
    result_string = string + "\n" + underline_string
    result_string = empty_lines_start + result_string + empty_lines_end
    return result_string

def sub_when_empty(string, empty_str="-"):
    if len(string) > 0:
        result_string = string
    else:
        result_string = empty_str
    return result_string

def tcs(string_1, string_2, sep=":", col_width=24, empty_str="-", par=True):
    # two-column-string
    result_string = f"{string_1+sep:{col_width}s}{sub_when_empty(string_2, empty_str=empty_str)}"
    if par:
        result_string += "\n"
    return result_string

def mcs(string_list, col_width=18, par=False):
    # multi-column-string
    result_string = ""
    for string in string_list:
        result_string += f"{string:{col_width}}"
    if par:
        result_string += "\n"
    return result_string



# def __str__(self):
#
#     title_string = underlined_string(self.name, n_empty_start=2)
#
#     n_prms = len(self._prm_names)
#     prms_string = underlined_string("Parameter overview", symbol="-")
#     prms_string += f"Number of parameters:   {n_prms}\n"
#     for group in self._prm_names_dict.keys():
#         prms_string += tcs(f'{group.capitalize()} parameters',
#                            self._prm_names_dict[group])
#
#     const_prms_str = underlined_string("Constant parameters", symbol="-")
#     w = len(max(self._const_dict.keys(), key=len)) + 2
#     for prm_name, prm_value in self._const_dict.items():
#         const_prms_str += tcs(prm_name, f"{prm_value:.2f}", col_width=w)
#
#     prior_str = underlined_string("Priors defined", symbol="-")
#     w = len(max(self._priors.keys(), key=len)) + 2
#
#     for prior_name, prior_obj in self._priors.items():
#         prior_str += tcs(prior_name, str(prior_obj), col_width=w)
#
#     full_string = title_string + prms_string + const_prms_str + prior_str
#
#     return full_string