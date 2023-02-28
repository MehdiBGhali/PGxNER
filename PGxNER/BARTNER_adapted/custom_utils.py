from fastNLP.core.field import Padder 
import numpy as np
from numbers import Number
from typing import Any
import torch

# Padding debug
class  DebugAutoPadder(Padder): 
    def __init__(self,pad_val = 0): 
        print("initiated")
        super().__init__(pad_val = pad_val)

    def __call__(self, contents, field_name, field_ele_dtype, dim):
        print(f"---------------------------- \n field name : {field_name} \n")
        print(f"content_to_pad : {contents} \n")
        print(f"content dims : {dim} \n")
        print(f"content elemnt type : {field_ele_dtype} \n")
        if field_ele_dtype:
            if dim > 3:
                print("Scenario 1 (dims too big)")
                return np.array(contents)
            if isinstance(field_ele_dtype, type) and \
                    (issubclass(field_ele_dtype, np.number) or issubclass(field_ele_dtype, Number)):
                print("Scenario 2 (number) \n")
                if dim == 0:
                    print("Scenario 2-a (number) of dim 0 \n")
                    array = np.array(contents, dtype=field_ele_dtype)
                elif dim == 1:
                    print("Scenario 2-b (number) of dim 1 \n")
                    max_len = max(map(len, contents))
                    array = np.full((len(contents), max_len), self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        array[i, :len(content_i)] = content_i
                elif dim == 2:
                    print("Scenario 2-c (number) of dim 2 \n")
                    max_len = max(map(len, contents))
                    max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                        content_i in contents])
                    array = np.full((len(contents), max_len, max_word_len), self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        for j, content_ii in enumerate(content_i):
                            array[i, j, :len(content_ii)] = content_ii
                else:
                    print("Scenario 2-d (number) of dim + \n")
                    shape = np.shape(contents)
                    if len(shape) == 4:  # 说明各dimension是相同的大小
                        array = np.array(contents, dtype=field_ele_dtype)
                    else:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                return array
            elif str(field_ele_dtype).startswith('torch'):
                print("Scenario 3 torch tensor \n")
                if dim == 0:
                    tensor = torch.tensor(contents).to(field_ele_dtype)
                elif dim == 1:
                    max_len = max(map(len, contents))
                    tensor = torch.full((len(contents), max_len), fill_value=self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        tensor[i, :len(content_i)] = content_i.clone().detach()
                elif dim == 2:
                    max_len = max(map(len, contents))
                    max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                        content_i in contents])
                    tensor = torch.full((len(contents), max_len, max_word_len), fill_value=self.pad_val,
                                        dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        for j, content_ii in enumerate(content_i):
                            tensor[i, j, :len(content_ii)] = content_ii.clone().detach()
                else:
                    shapes = set([np.shape(content_i) for content_i in contents])
                    if len(shapes) > 1:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                    shape = shapes.pop()
                    if len(shape) == 3:
                        tensor = torch.full([len(contents)] + list(shape), fill_value=self.pad_val,
                                            dtype=field_ele_dtype)
                        for i, content_i in enumerate(contents):
                            tensor[i] = content_i.clone().detach().to(field_ele_dtype)
                    else:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                return tensor
            else:
                print("Scenario 5 unaccounted elem type \n")
                return np.array(contents)  # 不进行任何操作
        else:
            print("Scenario 0 : None elements !!!!!!!!!!!!!!!!!!!!! \n")
            return np.array(contents, dtype = object)

class  adjusted_for_np_AutoPadder(Padder): 
    def __init__(self,pad_val = 0): 
        super().__init__(pad_val = pad_val)

    def __call__(self, contents, field_name, field_ele_dtype, dim):
        if field_ele_dtype:
            if dim > 3:
                return np.array(contents)
            if isinstance(field_ele_dtype, type) and \
                    (issubclass(field_ele_dtype, np.number) or issubclass(field_ele_dtype, Number)):
                if dim == 0:
                    array = np.array(contents, dtype=field_ele_dtype)
                elif dim == 1:
                    max_len = max(map(len, contents))
                    array = np.full((len(contents), max_len), self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        array[i, :len(content_i)] = content_i
                elif dim == 2:
                    max_len = max(map(len, contents))
                    max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                        content_i in contents])
                    array = np.full((len(contents), max_len, max_word_len), self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        for j, content_ii in enumerate(content_i):
                            array[i, j, :len(content_ii)] = content_ii
                else:
                    shape = np.shape(contents)
                    if len(shape) == 4:  # 说明各dimension是相同的大小
                        array = np.array(contents, dtype=field_ele_dtype)
                    else:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                return array
            elif str(field_ele_dtype).startswith('torch'):
                if dim == 0:
                    tensor = torch.tensor(contents).to(field_ele_dtype)
                elif dim == 1:
                    max_len = max(map(len, contents))
                    tensor = torch.full((len(contents), max_len), fill_value=self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        tensor[i, :len(content_i)] = content_i.clone().detach()
                elif dim == 2:
                    max_len = max(map(len, contents))
                    max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                        content_i in contents])
                    tensor = torch.full((len(contents), max_len, max_word_len), fill_value=self.pad_val,
                                        dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        for j, content_ii in enumerate(content_i):
                            tensor[i, j, :len(content_ii)] = content_ii.clone().detach()
                else:
                    shapes = set([np.shape(content_i) for content_i in contents])
                    if len(shapes) > 1:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                    shape = shapes.pop()
                    if len(shape) == 3:
                        tensor = torch.full([len(contents)] + list(shape), fill_value=self.pad_val,
                                            dtype=field_ele_dtype)
                        for i, content_i in enumerate(contents):
                            tensor[i] = content_i.clone().detach().to(field_ele_dtype)
                    else:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                return tensor
            else:
                return np.array(contents)  # 不进行任何操作
        else:
            return np.array(contents, dtype = object)


# FastNLP.field get type debug
def _get_ele_type_and_dim_debug(cell: Any, dim=0):
    if isinstance(cell, (str, Number, np.bool_)):
        if hasattr(cell, 'dtype'):
            print(f"dim 0, dtype trouvé ({cell.dtype.type})")
            return cell.dtype.type, dim
        print(f"dim 0, type trouvé ({type(cell)})")
        return type(cell), dim
    elif isinstance(cell, list):
        print(f"list encountered : {cell}")
        dim += 1
        print(f"new dim : {dim}")
        res = [_get_ele_type_and_dim_debug(cell_i, dim) for cell_i in cell]
        types = set([i for i, j in res])
        dims = set([j for i, j in res])
        if len(types) > 1:
            raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
        elif len(types) == 0:
            raise SetInputOrTargetException("Empty value encountered.")
        if len(dims) > 1:
            raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
        return types.pop(), dims.pop()
    elif isinstance(cell, torch.Tensor):
        print("torch encountered")
        return cell.dtype, cell.dim() + dim  # 如果是torch.mean的结果是0
    elif isinstance(cell, np.ndarray):
        print("np array encountered")
        if cell.dtype != np.dtype('O'):  # 如果不是object的话说明是well-formatted的了
            return cell.dtype.type, cell.ndim + dim  # dtype.type返回的会是np.int32, np.float等
        # 否则需要继续往下iterate
        dim += 1
        res = [_get_ele_type_and_dim_debug(cell_i, dim) for cell_i in cell]
        types = set([i for i, j in res])
        dims = set([j for i, j in res])
        if len(types) > 1:
            raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
        elif len(types) == 0:
            raise SetInputOrTargetException("Empty value encountered.")
        if len(dims) > 1:
            raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
        return types.pop(), dims.pop()
    else:  # 包含tuple, set, dict以及其它的类型
        raise SetInputOrTargetException(f"Cannot process type:{type(cell)}.")

def _check_dtype_and_ndim_debug(field, only_check_1st_ins_dim_debug_type=True):
    cell_0 = field.content[0]
    index = 0
    try:
        type_0, dim_0 = _get_ele_type_and_dim_debug(cell_0)
        print(type_0,dim_0)
        if not only_check_1st_ins_dim_debug_type:
            for cell in field.content[1:]:
                index += 1
                type_i, dim_i = _get_ele_type_and_dim_debug(cell)
                if type_i != type_0:
                    raise SetInputOrTargetException(
                        "Type:{} in index {} is different from the first element with type:{}."
                        ".".format(type_i, index, type_0))
                if dim_0 != dim_i:
                    raise SetInputOrTargetException(
                        "Dimension:{} in index {} is different from the first element with "
                        "dimension:{}.".format(dim_i, index, dim_0))
        field._cell_ndim = dim_0
        field.dtype = type_0
    except SetInputOrTargetException as e:
        e.index = index
        raise e
