"""
Neural Logic Machine (NLM) implementation by Carlos Núñez Molina.
This implemenation has a unique optimization that reduces computation.

NLM takes a list of tensors and returns a list of tensors.
Each tensor in the list contains all predicates of a particular arity.
The input dimensions are

  [B, F_0]                  where
  [B, O, F_1]                 B : batch size,
  [B, O, O, F_2]              O : number of objects,
  [B, O, O, O, F_3] ...       Fi: number of predicates of arity i.
  [B, O, O,  ..., F_A]        A : maximum arity.

When out_features is a list [Q_0, Q_1, ...Q_A], the output dimensions are

  [B, Q_0]
  [B, O, Q_1]
  [B, O, O, Q_2]
  [B, O, O, O, Q_3]
  [B, O, O,  ..., Q_A].

Standard NLM works by expansion, reduction and permutation.
For example, for arity 2, we get

  [B, O, O, F_1] : expand -\                        permute                       linear
  [B, O, O, F_2]          -+-> [B, O, O, F_1+F_2+F_3] -> [B, O, O, 2*(F_1+F_2+F_3)] -> [B, O, O, Q_2].
  [B, O, O, F_3] : reduce -/

An issue with a standard NLM is that
it requires a large amount of zero-padding for handling different number of objects.

Imagine concatenating two binary NLM representations as follows.

  [B1, 10, 10, 7] and [B2, 20, 20, 7] -> [B1+B2, 20, 20, 7]

Notice that B1*10*10*3*7 floats are zero-filled.

"""

from typing import Union, Literal, Optional
ResidualType = Literal[None, "input", "all"]
Activation = Literal["sigmoid", "relu"]

import torch
from torch import nn
import numpy as np
import itertools
import functools

from pddlsl.constants import inf, ninf, EPS

def mask_invalid_pos(X:torch.Tensor, num_obj_list:list[int], value:float):
    """
    Masks the invalid positions of the input tensor X, of shape [B, N*arity, P]. Invalid positions are those
    corresponding to non-existing objects for each sample in the batch.
    For example, if the number of objects for the i-th element is n=2, the positions X[i,n:,...,n:,:] are invalid,
    as they simply correspond to padding and they should be ignored by the NLM.
    In order to do that, we mask those values using this method.
    """
    with torch.no_grad():
        arity = X.dim() - 2
        B = X.shape[0] # batch size
        N = X.shape[1] # maximum number of objects
        P = X.shape[-1] # number of predicates
        device = X.device # Create all tensors in GPU

        # Obtain index tensor
        if arity == 0: # For arity 0, no masking is needed (since there are no objects)
            return X

        elif arity == 1:
            t = torch.arange(N, device=device)

        elif arity == 2:
            t1 = torch.arange(N, device=device)
            t2 = t1.unsqueeze(0).T
            t = torch.maximum(t1, t2)

        elif arity == 3:
            t_a = torch.arange(N, device=device)
            t0 = t_a.unsqueeze(1).unsqueeze(2)
            t1 = t_a.unsqueeze(0).unsqueeze(2)
            t2 = t_a.unsqueeze(0).unsqueeze(1)
            t = torch.maximum(torch.maximum(t0, t1), t2)

        else:
            raise NotImplementedError("Right now, we can only mask tensors corresponding to arities between 1 and 3")

        # Expand index tensor from shape [N*arity] to [B,N*arity,P]
        t_P = t.unsqueeze(-1).expand( *((-1,)*arity+(P,)) ) # For arity 3, this would be .expand(-1,-1,-1,P)
        index_tensor = t_P.unsqueeze(0).expand( *((B,)+(-1,)*(arity+1)) ) # For arity 3, this would be .expand(B,-1,-1,-1,-1)

        # Obtain the mask
        # It equals True for invalid positions, i.e., those corresponding to non-existing objects (padded positions)
        # If num_obj_list[i]=2, then X[i,2:,...,2:,:] equals True (i.e., all positions corresponding to objects whose index is
        # larger than 1 equal True), where the number of "2:" corresponds to the arity
        mask = index_tensor >= torch.tensor(num_obj_list, device=device).view( (-1,)+(1,)*(arity+1) )

    # Mask out invalid positions by setting them to val parameter
    # We create a new tensor instead of modifying it in-place
    masked_tensor = X.masked_fill(mask, value)

    return masked_tensor

class _InferenceMLP(nn.Module):
    def __init__(self, arity:int, hidden_size:int, output_size:int, batchnorm:bool, activation:Activation, activate_output:bool=True):
        super().__init__()

        self.arity = arity
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.activate_output = activate_output
        self.batchnorm = batchnorm

        activation_class = {'sigmoid':nn.Sigmoid, 'relu':nn.ReLU}[activation]

        if hidden_size < 1: # Do not use hidden layer
            self.layers = nn.Sequential(
                nn.LazyLinear(output_size, bias=not batchnorm),
                *([nn.BatchNorm1d(output_size)] if batchnorm else []),
                *([activation_class()] if activate_output else []),
            )
        else: # Use hidden layer
            self.layers = nn.Sequential(
                nn.LazyLinear(hidden_size, bias=not batchnorm),
                *([nn.BatchNorm1d(hidden_size)] if batchnorm else []),
                activation_class(),
                nn.Linear(hidden_size, output_size, bias=not batchnorm),
                *([nn.BatchNorm1d(output_size)] if batchnorm else []),
                *([activation_class()] if activate_output else []),
            )
        pass


    def forward(self, X:torch.Tensor):
        """
        The shape of the input tensor X is [B, N, N, ..., P], where
        B is the batch size, N the maximum number of objects, the number of N
        is equal to the predicate arity and P is the number of predicates for
        this arity (equal to the input size of the MLP).
        The MLP computes the output for every NxN...xN object combination
        in each batch sample, even if the particular sample had fewer objects
        n < N. Example N=4, n=2: the MLP computes an output for the position
        [3,3,p] even though the sample has no object with index 3. These outputs
        will be ignored by the rest of the NLM operations and are only done
        for efficient purposes, i.e., padding and stacking the tensors in order
        to execute all operations in parallel for the entire batch.
        """
        mlp_output = self.layers(X) # Output shape: [B, N, N, ..., self.output_size]

        return mlp_output

class _NLM_Layer(nn.Module):

    """
    Creates the NLM layer and initializes the inference MLPs. There is a different inference MLP
    to compute each output predicate arity. For example: if out_features = [0, 8, 4, 0] (only compute
    output unary and binary predicates), then there are two different inference MLPs for this layer.

    @in_features List with the number of input predicates of each arity, in ascending order. It is equal to
                                out_features for the previous layer.
                                Example: [8, 8, 4, 0] -> The previous NLM layer's output consisted of 8 nullary predicates,
                                                          8 unary predicates and 4 binary predicates.
    @out_features List with the number of output predicates of each arity, in ascending order.
    @mlp_hidden_features Units in the hidden layer of all the inference MLPs. If 0, the inference MLPs have no hidden layer.
    @activate_output Whether the MLPs should apply an activation function to the output of the last layer.
    @exclude_self If True, the NLM ignores tensor positions corresponding to repeated indexes (e.g., [5][5][3] or [2][2][0][1])
                  when performing the reduce operation
    @reduce_masks_cache A dictionary with the masks used for the reduce operation when exclude_self=True.
                        This way, we can avoid recomputing the masks.
    """
    def __init__(self, out_features:list[int], mlp_hidden_features:int, batchnorm, activation,
                 activate_output, exclude_self=True,
                 reduce_masks_cache=None):
        super().__init__()

        self.out_features = out_features
        self.exclude_self = exclude_self
        self.max_arity = out_features.shape[0]-1
        self.reduce_masks_cache = reduce_masks_cache

        if exclude_self and self.max_arity > 3:
            raise NotImplementedError("Can't use exclude_self=True for NLMs with breadth > 3")

        if exclude_self and reduce_masks_cache is None:
            raise Exception("If exclude_self is True, we need to use reduce_masks_cache (cannot be None)")

        self._mlps = nn.ModuleList([_InferenceMLP(i, mlp_hidden_features, out_features[i], batchnorm, activation, activate_output) \
                                   if out_features[i] > 0 else None \
                                   for i in range(len(out_features))])


    def _compute_mask(self, arity, num_objs, device):
        assert arity > 0, "We can't reduce nullary predicates"
        assert arity < 4, "No support for tensors of arity bigger than 3"

        # Unary predicates -> all the positions are valid (since there are no repeated objects)
        if arity == 1:
            return torch.ones(num_objs, device=device).unsqueeze(0).unsqueeze(-1)
        # Binary predicates -> mask has 1 in every position except for the diagonal
        elif arity == 2:
            return (1 - torch.eye(num_objs, device=device)).unsqueeze(0).unsqueeze(-1)
        # Ternary predicates -> mask has 1 in every position except where two variables are the same (x==y, x==z or y==z)
        else:
            binary_mask = 1 - torch.eye(num_objs, device=device)
            ternary_mask = functools.reduce(lambda a,b: a*b, [torch.unsqueeze(binary_mask, dim=dim) for dim in range(3)])
            return ternary_mask.unsqueeze(0).unsqueeze(-1)

    def _append_reduce_masks_to_cache(self, num_objs, device):
        with torch.no_grad():
            self.reduce_masks_cache[num_objs] = []
            for r in range(1, self.max_arity+1):
                self.reduce_masks_cache[num_objs].append([])
                self.reduce_masks_cache[num_objs][-1].append(self._compute_mask(r, num_objs, device))
                self.reduce_masks_cache[num_objs][-1].append(1-self.reduce_masks_cache[num_objs][-1][0]) # Also store 1-m

    def _get_mask(self, ind, arity, num_objs, device):
        """
        Obtains a mask for duplicate arguments needed by the reduce operation when exclude_self=True.
        If ind==0, we return the reduce masks for the "max" reduction and, if ind==1, we return the masks for the
        "min" reduction (corresponding to 1-m).
        In order to save up computation time, we save the computed masks in the cache "self.reduce_masks_cache".
        """
        # Retrieve the mask from the cache if it exists. If not, compute the mask and store it in cache for later use
        if num_objs not in self.reduce_masks_cache:
            self._append_reduce_masks_to_cache(num_objs, device)

        return self.reduce_masks_cache[num_objs][arity-1][ind]

    def _expand(self, X, N):
        """
        We transform a tensor of predicates of arity a into predicate a+1, by adding an additional variable to each predicate
        and copying the predicate values along the new axis (corresponding to the new variable).

        Example (2 predicates of arity 1, instantiated on 3 objects): input shape = [B, 3, 2] -> output shape = [B, 3, 3, 2]

        @X A tensor containing the tensors of the corresponding arity for each element in the batch, of shape [B, N*arity, P].
        @N Maximum number of objects, used for expanding the tensor
        @num_obj_list A list where each element contains the number of objects of the sample X[i]
        """
        dim = X.dim()
        # expanded_tensor = X.unsqueeze(-2).repeat( (1,)*(dim-1) + (N,) + (1,) )
        # We use expand instead of repeat because it is more efficient, but expand() does duplicate references!
        expanded_tensor = X.unsqueeze(-2).expand( (-1,)*(dim-1) + (N,) + (-1,) )

        return expanded_tensor

    def _reduce(self, X, num_obj_list, reduce_type):
        """
        We transform a tensor of predicates of arity a into predicate a-1, by deleting the last variable by taking the maximum
        or minimum element along the corresponding axis.

        Example (2 predicates of arity 2, instantiated on 3 objects): input shape = [3, 3, 2] -> output shape = [3, 2]

        @X A tensor containing the tensors of the corresponding arity for each element in the batch, of shape [B, N*arity, P].
        @reduce_type Either 'min' (corresponding to "forall") or 'max' (corresponding to "exists")
        """
        # If reduce_type=="min", we calculate the min along the -2 axis, else we take the maximum
        data_device = X.device
        N = X.shape[1]
        arity = X.dim()-2
        assert arity > 0, "The arity of the tensor Xs must be 1 or larger (nullary predicates cannot be reduced)"

        # If self.exclude_self is True, we mask out those tensor positions corresponding to repeated objects
        if self.exclude_self:
            if reduce_type == 'max':
                X_masked = X * self._get_mask(0, arity, N, data_device)
            else: # 'min'
                X_masked = X * self._get_mask(0, arity, N, data_device) + \
                               self._get_mask(1, arity, N, data_device) # _get_mask(1,...) obtains the 1-m mask
        else:
            X_masked = X

        # Before applying the reduce operation, we need to mask out the tensor positions corresponding to invalid objects
        # (i.e., padded values). For min, values in these positions should be inf and, for max, they should be -inf, so that
        # they don't affect the result of the reduction operation in the invalid positions.
        #
        # Note: Do not use float("inf"); use a finite value, otherwise it causes NaN gradients
        # Note: Xs is not modified
        X_masked_invalid = mask_invalid_pos(X_masked,
                                            num_obj_list,
                                            {"min":inf, "max":ninf}[reduce_type])

        # Apply the actual reduce operation
        # We reduce the last object dimension (i.e., -2)
        if reduce_type == 'max':
            reduced_tensor = torch.amax(X_masked_invalid, -2)
        elif reduce_type == 'min':
            reduced_tensor = torch.amin(X_masked_invalid, -2)
        else:
            raise NotImplementedError("Right now, only 'max' and 'min' reductions are implemented")

        return reduced_tensor

    def _permute(self, X):
        """
        We return a tensor with all the possible permutations of the input tensor's axes that index objects.
        The extra tensors are appended as if they were additional predicates.

        Example: input shape = [B, 3, 3, 2] -> ouput_shape = [B, 3, 3, 4]
        Example2: input shape = [B, 3, 3, 3, 1] -> ouput_shape = [B, 3, 3, 3, 6]

        <Note>: the new tensors corresponding to permutations are views of the original tensor, in order to reduce
          memory consumption. However, this means they share the reference!

        @X A tensor containing the tensors of the corresponding arity for each element in the batch, of shape [B, N*arity, P].
        """
        dim = X.dim()

        if dim < 4: # dim=2 -> nullary predicates, dim=3 -> unary predicates : they do not need permutations
            return X

        obj_axes = range(1, dim-1) # Indexes of axes that correspond to objects in the tensor (e.g.: [1, 2] for binary predicates)
        obj_axes_permutations = itertools.permutations(obj_axes)
        last_axis = (dim-1,) # Axis corresponding to the predicates, which is NOT permuted

        # For each tensor in X, concatenate along the predicate dimension (torch.cat(..., dim=-1)) all the possible permutations
        # of the axes that index objects (obj_axes_permutations)
        permuted_tensor = torch.cat([X.permute((0,) + perm + last_axis) for perm in obj_axes_permutations], dim=-1)

        return permuted_tensor


    def forward(self, X:list[Optional[torch.Tensor]], num_obj_list:list[int]) -> list[torch.Tensor]:
        """
        It receives a list @input_tensors_list with the input tensors (corresponding to the output tensors of the previous NLM layer)
        and returns a list with the output tensors of the current NLM layer.
        If there are no input tensors for arity r, then @input_tensors_list[r] must be None.

        @X A list with all the tensors for all the samples in the batch.
           X[r] is a tensor of shape [B, N*r, P] containing the predicates of arity r for all the samples in the batch
        @num_obj_list A list where each element contains the number of objects of the sample X[r][i] for any arity r.
        """

        # Obtain the maximum number of objects N
        X_not_None = [x for x in X[1:] if x is not None] # We skip the nullary predicates, since they contain no dimension N
        assert len(X_not_None) > 0, "X does not contain predicates of arity larger than 0!"
        N = X_not_None[0].shape[1]
        max_arity = len(X)-1

        mlp_inputs = []

        for r in range(max_arity+1):
            # If we do not need the output predicates for arity r, we skip this arity
            if self.out_features[r] > 0:
                cat_tensors = []
                # Expand arity r-1
                if r > 0 and X[r-1] is not None:
                    cat_tensors.append(self._expand(X[r-1], N))
                # Tensor arity r
                if X[r] is not None:
                    cat_tensors.append(X[r])
                # Reduce arity r+1, with min and max
                if r < max_arity and X[r+1] is not None:
                    cat_tensors.append(self._reduce(X[r+1], num_obj_list, "min"))
                    cat_tensors.append(self._reduce(X[r+1], num_obj_list, "max"))
                # Concatenate the tensors
                cat_tensors = torch.cat(cat_tensors, dim=-1)
                # Permute the tensors (for arity < 2, self._permute() simply returns the tensor without permuting anything)
                permuted_tensors = self._permute(cat_tensors)
                mlp_inputs.append(permuted_tensors)
            else:
                mlp_inputs.append(None) # Append None if we do not need the complete input tensors for arity r

        # Obtain the output tensors by applying the MLP to the real input tensors
        output_tensors_list = []
        out_features_len = len(self.out_features)

        for r in range(out_features_len):
            if self.out_features[r] == 0: # We do not need to compute the output predicates for this arity
                output_tensors_list.append(None)
            else:
                output_tensors_list.append(self._mlps[r](mlp_inputs[r]))

        return output_tensors_list




class NLMCarlosV2(nn.Module):
    def __init__(self,
                 hidden_features:list[list[int]],
                 out_features:list[int],
                 mlp_hidden_features:int,
                 residual : ResidualType = None,
                 exclude_self : bool = True,
                 batchnorm : bool = False,
                 activation : Activation = "sigmoid",
                 mask_value : float = ninf):
        super().__init__()

        all_output_features = np.array(hidden_features+[out_features], dtype=int)
        depth, max_arity = all_output_features.shape[0], all_output_features.shape[1]-1

        self.residual = residual
        self.mask_value = mask_value

        # Cache that contains the masks used by the _reduce() method when exclude_self=True
        # This cache is shared between all the _NLM_Layer objects in order to save up memory and computation time
        # reduce_masks_cache[n][r-1][0] stores the reduce masks employed for the 'max' operation corresponding to
        # arity=r and num_objs=n
        # reduce_masks_cache[n][r-1][1] stores the reduce masks for the 'min' operation, equal to 1-reduce_masks_cache[n][r-1][0]
        reduce_masks_cache = dict()

        self.layers = nn.ModuleList([_NLM_Layer(all_output_features[i],
                                                mlp_hidden_features,
                                                activation=activation,
                                                batchnorm            = (i != depth-1 and batchnorm),
                                                activate_output      = (i != depth-1),
                                                exclude_self = exclude_self,
                                                reduce_masks_cache = reduce_masks_cache) \
                                     for i in range(depth)])

        self.num_layers = len(self.layers)


    def forward(self, xs:list[Optional[torch.Tensor]], num_objs:Union[int, list[int]]):
        # Detect whether @xs corresponds to a single sample or a batch of samples
        # If it corresponds to a single sample, we encode it as a single-sample batch
        if type(num_objs) == int: # @xs corresponds to a single sample
            xs = [x.unsqueeze(0) if x is not None else None for x in xs]
            num_objs = [num_objs]

        def cat(args,dim=-1):
            tmp = []
            for arg in args:
                if arg is not None:
                    tmp.append(arg)
            if len(tmp) == 1:
                return tmp[0]
            else:
                return torch.cat(tmp,dim=dim)

        xss = [xs]
        for i in range(self.num_layers):
            if self.residual == "input" and i > 0: # concat input layer
                xss.append(
                    self.layers[i](
                        [cat([x,y],dim=-1) for x,y in zip(xss[-1], xss[0])],
                        num_objs))
            elif self.residual == "all" and i > 0: # concat all previous layers
                xss.append(
                    self.layers[i](
                        [cat(xss_r,dim=-1) for xss_r in zip(*xss)],
                        num_objs))
            else:
                xss.append(
                    self.layers[i](
                        xss[-1],
                        num_objs))

        X_curr_masked = [mask_invalid_pos(t, num_objs, self.mask_value) if t is not None else None for t in xss[-1]]

        return X_curr_masked



if __name__ == "__main__":

    import pddlsl.stacktrace as stacktrace
    import random
    try:
        O = 13
        B = 17
        Q = 19
        Os = [ random.randint(1, O) for _ in range(B) ]
        x = [
            torch.zeros((B, 2)),
            torch.zeros((B, O, 3)),
            torch.zeros((B, O, O, 5)),
            torch.zeros((B, O, O, O, 7)),
            # max arity is 3
        ]
        for residual in [None, "input", "all"]:
            L = NLMCarlosV2(hidden_features = [[Q,Q,Q,Q],[Q,Q,Q,Q]],
                            out_features = [2, 0, 0, 0],
                            mlp_hidden_features = 0,
                            residual = residual)
            y = L(x,Os)
            for i, yi in enumerate(y):
                print(f"residual {residual} output arity {i}: shape {yi.size() if yi is not None else None}")
            print(L)



    except:
        stacktrace.format()
