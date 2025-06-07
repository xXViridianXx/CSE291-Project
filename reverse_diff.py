import inspect
import random
import string
import autodiff
import irmutator
import _asdl.loma as loma_ir
import ir
ir.generate_asdl_file()


DEBUG = True
# From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits


def random_id_generator(size=6, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def debug_node(node):
    if not DEBUG:
        return
    label = f"In {inspect.currentframe().f_back.f_code.co_name}"
    border = "-" * len(label)
    print(f"\n{border}")
    print(f"In {inspect.currentframe().f_back.f_code.co_name}")
    print(node)
    print(f"{border}\n")
            
def get_type(self, node):
    match node:
        case loma_ir.Var():
            return node.t
        case loma_ir.Struct():
            return self.get_type(node.struct)
        case loma_ir.Array():
            return self.get_type(node.t)
        case _:
            assert False
            
def reverse_diff(diff_func_id: str,
                 structs: dict[str, loma_ir.Struct],
                 funcs: dict[str, loma_ir.func],
                 diff_structs: dict[str, loma_ir.Struct],
                 func: loma_ir.FunctionDef,
                 func_to_rev: dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply reverse differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', reverse_diff() should return
        def d_square(x : In[float], _dx : Out[float], _dreturn : float):
            _dx = _dx + _dreturn * x + _dreturn * x

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_rev - mapping from primal function ID to its reverse differentiation
    """

    # Some utility functions you can use for your homework.
    def type_to_string(t):
        match t:
            case loma_ir.Int():
                return 'int'
            case loma_ir.Float():
                return 'float'
            case loma_ir.Array():
                return 'array_' + type_to_string(t.t)
            case loma_ir.Struct():
                return t.id
            case _:
                assert False

    def var_to_differential(expr, var_to_dvar):
        match expr:
            case loma_ir.Var():
                return loma_ir.Var(var_to_dvar[expr.id], t=expr.t)
            case loma_ir.ArrayAccess():
                return loma_ir.ArrayAccess(
                    var_to_differential(expr.array, var_to_dvar),
                    expr.index,
                    t=expr.t)
            case loma_ir.StructAccess():
                return loma_ir.StructAccess(
                    var_to_differential(expr.struct, var_to_dvar),
                    expr.member_id,
                    t=expr.t)
            case _:
                assert False

    def assign_zero(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t=m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t=m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert False
        
    def assign_zero_hw3(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Array():
                return assign_zero(target.t)
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t=m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t=m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert False

    def accum_deriv(target, deriv, overwrite):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                if overwrite:
                    return [loma_ir.Assign(target, deriv)]
                else:
                    
                    return [loma_ir.CallStmt(loma_ir.Call("atomic_add", [target, deriv]))]
                    return [loma_ir.Assign(target,
                                           loma_ir.BinaryOp(loma_ir.Add(), target, deriv))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t=m.t)
                    deriv_m = loma_ir.StructAccess(
                        deriv, m.id, t=m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t=m.t.t)
                            deriv_m = loma_ir.ArrayAccess(
                                deriv_m, loma_ir.ConstInt(i), t=m.t.t)
                            stmts += accum_deriv(target_m, deriv_m, overwrite)
                return stmts
            case _:
                assert False

    def check_lhs_is_output_arg(lhs, output_args):
        match lhs:
            case loma_ir.Var():
                return lhs.id in output_args
            case loma_ir.StructAccess():
                return check_lhs_is_output_arg(lhs.struct, output_args)
            case loma_ir.ArrayAccess():
                return check_lhs_is_output_arg(lhs.array, output_args)
            case _:
                assert False

    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    # Furthermore, it normalizes all Assign statements
    # with a function call
    # z = f(...)
    # into a declaration followed by an assignment
    # _tmp : [z's type]
    # _tmp = f(...)
    # z = _tmp
    class CallNormalizeMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            debug_node(node)
            self.tmp_count = 0
            self.tmp_declare_stmts = []
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)

            new_body = self.tmp_declare_stmts + new_body

            return loma_ir.FunctionDef(
                node.id, node.args, new_body, node.is_simd, node.ret_type, lineno=node.lineno)

        def mutate_return(self, node):
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(
                val,
                lineno=node.lineno)]

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Declare(
                node.target,
                node.t,
                val,
                lineno=node.lineno)]

        def mutate_assign(self, node):
            self.tmp_assign_stmts = []
            target = self.mutate_expr(node.target)
            self.has_call_expr = False
            val = self.mutate_expr(node.val)
            if self.has_call_expr:
                # turn the assignment into a declaration plus
                # an assignment
                self.tmp_count += 1
                tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                self.tmp_count += 1
                self.tmp_declare_stmts.append(loma_ir.Declare(
                    tmp_name,
                    target.t,
                    lineno=node.lineno))
                tmp_var = loma_ir.Var(tmp_name, t=target.t)
                assign_tmp = loma_ir.Assign(
                    tmp_var,
                    val,
                    lineno=node.lineno)
                assign_target = loma_ir.Assign(
                    target,
                    tmp_var,
                    lineno=node.lineno)
                return self.tmp_assign_stmts + [assign_tmp, assign_target]
            else:
                return self.tmp_assign_stmts + [loma_ir.Assign(
                    target,
                    val,
                    lineno=node.lineno)]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            
            print(f"pachurisue: {loma_ir.CallStmt(call, lineno=node.lineno)}")
            return self.tmp_assign_stmts + [loma_ir.CallStmt(
                call,
                lineno=node.lineno)]

        def mutate_call(self, node):
            debug_node(node)
            self.has_call_expr = True
            new_args = []
            for arg in node.args:
                if not isinstance(arg, loma_ir.Var) and \
                        not isinstance(arg, loma_ir.ArrayAccess) and \
                        not isinstance(arg, loma_ir.StructAccess):
                    arg = self.mutate_expr(arg)
                    tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                    self.tmp_count += 1
                    tmp_var = loma_ir.Var(tmp_name, t=arg.t)
                    self.tmp_declare_stmts.append(loma_ir.Declare(
                        tmp_name, arg.t))
                    self.tmp_assign_stmts.append(loma_ir.Assign(
                        tmp_var, arg))
                    new_args.append(tmp_var)
                else:
                    new_args.append(arg)
            print(f"Normalize mutate call: {loma_ir.Call(node.id, new_args, t=node.t)}")
            return loma_ir.Call(node.id, new_args, t=node.t)

    class ForwardPassMutator(irmutator.IRMutator):
        def __init__(self, output_args):
            self.output_args = output_args
            self.cache_vars_list = {}
            self.var_to_dvar = {}
            self.type_cache_size = {}
            self.type_to_stack_and_ptr_names = {}
            self.in_call_stmt = False
            self.while_stack = 0
            self.tmp_declares = []
            
        def mutate_return(self, node):
            return []

        def mutate_declare(self, node):
            # For each declaration, add another declaration for the derivatives
            # except when it's an integer
            if node.t != loma_ir.Int():
                dvar = '_d' + node.target + '_' + random_id_generator()
                self.var_to_dvar[node.target] = dvar
                return [node, loma_ir.Declare(
                    dvar,
                    node.t,
                    lineno=node.lineno)]
            else:
                return node

        def mutate_assign(self, node):
            debug_node(node)
            if check_lhs_is_output_arg(node.target, self.output_args):
                return []

            # y = f(x0, x1, ..., y)
            # we will use a temporary array _t to hold variable y for later use:
            # _t[stack_pos++] = y
            # y = f(x0, x1, ..., y)
            assign_primal = loma_ir.Assign(
                node.target,
                self.mutate_expr(node.val),
                lineno=node.lineno)
            # backup
            t_str = type_to_string(node.val.t)
            
            if t_str in self.type_to_stack_and_ptr_names:
                stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
            else:
                random_id = random_id_generator()
                stack_name = f'_t_{t_str}_{random_id}'
                stack_ptr_name = f'_stack_ptr_{t_str}_{random_id}'
                
                self.type_to_stack_and_ptr_names[t_str] = (
                    stack_name, stack_ptr_name)
                
            
            stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
            cache_var_expr = loma_ir.ArrayAccess(
                loma_ir.Var(stack_name),
                stack_ptr_var,
                t=node.val.t)
            
            cache_primal = loma_ir.Assign(cache_var_expr, node.target)
            stack_advance = loma_ir.Assign(stack_ptr_var,
                                           loma_ir.BinaryOp(loma_ir.Add(), stack_ptr_var, loma_ir.ConstInt(1)))

            if node.val.t in self.cache_vars_list:
                self.cache_vars_list[node.val.t].append(
                    (cache_var_expr, node.target))
            else:
                self.cache_vars_list[node.val.t] = [
                    (cache_var_expr, node.target)]
            if node.val.t in self.type_cache_size:
                self.type_cache_size[node.val.t] += 1
            else:
                self.type_cache_size[node.val.t] = 1
            return [cache_primal, stack_advance, assign_primal]
        
        def mutate_while(self, node):
            self.while_stack += 1

            debug_node(node)
            
            return self.while_forward(node, ctr_id=0) 
            
            return super().mutate_while(node)
        
        def while_forward(self, node, ctr_id):
            
            
            ctr_name, ctr = f"ctr{ctr_id}", None
            ctr_ptr_name, ctr_stack_ptr = None, None
            if ctr_id == 0:
                ctr = loma_ir.Declare(ctr_name, loma_ir.Int(), loma_ir.ConstInt(0))
            
            if ctr_id > 0:
                ctr = loma_ir.Declare(ctr_name, loma_ir.Array(loma_ir.Int(), self.while_stack))
                
                ctr_ptr_name = f"{ctr_name}_ptr"
                ctr_stack_ptr = loma_ir.Declare(ctr_ptr_name, loma_ir.Int(), loma_ir.ConstInt(0))

            self.while_stack *= node.max_iter
                
            self.tmp_declares.append(ctr)
            
            if ctr_stack_ptr: self.tmp_declares.append(ctr_stack_ptr)
            
            
            body = []
            for stmt in node.body:
                if isinstance(stmt, loma_ir.While):
                    body += self.while_forward(stmt, ctr_id + 1) 
                else: body += self.mutate_stmt(stmt)
            out = []
            
            if ctr_id > 0: 
                # body += loma_ir.Assign(stack_ptr_var, loma_ir.BinaryOp(loma_ir.Add(), stack_ptr_var, loma_ir.ConstInt(1)))
                index_location = loma_ir.ArrayAccess(loma_ir.Var(ctr_name), loma_ir.Var(ctr_ptr_name))
                body.append(loma_ir.Assign(index_location, loma_ir.BinaryOp(loma_ir.Add(), index_location, loma_ir.ConstInt(1))))
                
                out.append(loma_ir.Assign(loma_ir.Var(ctr_ptr_name), loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var(ctr_ptr_name), loma_ir.ConstInt(1))))
            else:
                print("garchomp")
                body.append(loma_ir.Assign(loma_ir.Var(ctr_name), loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var(ctr_name), loma_ir.ConstInt(1))))
            
            while_node = loma_ir.While(node.cond, node.max_iter, body)
            
            out = [while_node] + out 
            return out

            
            
        def mutate_call_stmt(self, node):
            if node.call.id == 'atomic_add':
                return []
                return node
            debug_node(node)
            func_def = funcs[node.call.id]
            result = []
            print(self.output_args)
            print(func_def)
            
            for call_arg, func_arg in zip(node.call.args, func_def.args):
                if isinstance(func_arg.i, loma_ir.In): continue
                func_out_is_call_stmt_out = check_lhs_is_output_arg(call_arg, self.output_args)
                if func_out_is_call_stmt_out: return []
                
                t_str = type_to_string(call_arg.t)
            #     print("torchic")
            #     print(t_str)

                if t_str in self.type_to_stack_and_ptr_names:
                    stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                else:
                    random_id = random_id_generator()
                    stack_name = f'_t_{t_str}_{random_id}'
                    stack_ptr_name = f'_stack_ptr_{t_str}_{random_id}'
                    self.type_to_stack_and_ptr_names[t_str] = (
                        stack_name, stack_ptr_name)

                stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                cache_var_expr = loma_ir.ArrayAccess(
                    loma_ir.Var(stack_name),
                    stack_ptr_var,
                    t=call_arg.t)

                # var = loma_ir.Var(id=arg.id, t=arg.t)
                cache_primal = loma_ir.Assign(cache_var_expr, call_arg)
                stack_advance = loma_ir.Assign(stack_ptr_var,
                                               loma_ir.BinaryOp(loma_ir.Add(), stack_ptr_var, loma_ir.ConstInt(1)))

                if call_arg.t in self.cache_vars_list:
                    self.cache_vars_list[call_arg.t].append((cache_var_expr, call_arg))
                else:
                    self.cache_vars_list[call_arg.t] = [(cache_var_expr, call_arg)]
                if call_arg.t in self.type_cache_size:
                    self.type_cache_size[call_arg.t] += 1
                else:
                    self.type_cache_size[call_arg.t] = 1

                result.extend([cache_primal, stack_advance])
            result.append(node)
            return result
    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.

    class RevDiffMutator(irmutator.IRMutator):

        def make_arg(self, o_arg):
            if not isinstance(o_arg, loma_ir.Var):
                return
            d_id = "_d" + o_arg.id + random_id_generator() + "_"
            d_arg = loma_ir.Var(id=d_id, t=o_arg.t)
            return d_arg



        def mutate_function_def(self, node):
            random.seed(hash(node.id))
            # Each input argument is followed by an output (the adjoint)
            # Each output is turned into an input
            # The return value turn into an input
            self.var_to_dvar = {}
            new_args = []
            self.output_args = set()

            node = CallNormalizeMutator().mutate_function_def(node)
            
            for arg in node.args:
                if arg.i == loma_ir.In():
                    new_args.append(arg)
                    dvar_id = '_d' + arg.id + '_' + random_id_generator()
                    new_args.append(loma_ir.Arg(
                        dvar_id, arg.t, i=loma_ir.Out()))
                    self.var_to_dvar[arg.id] = dvar_id
                else:
                    assert arg.i == loma_ir.Out()
                    self.output_args.add(arg.id)
                    new_args.append(loma_ir.Arg(arg.id, arg.t, i=loma_ir.In()))
                    self.var_to_dvar[arg.id] = arg.id

            if node.ret_type is not None:
                self.return_var_id = '_dreturn_' + random_id_generator()
                new_args.append(loma_ir.Arg(self.return_var_id,
                                node.ret_type, i=loma_ir.In()))
            
            # Forward pass
            fm = ForwardPassMutator(self.output_args)
            forward_body = node.body
            mutated_forward = [fm.mutate_stmt(
                fwd_stmt) for fwd_stmt in forward_body]
            mutated_forward = irmutator.flatten(mutated_forward)
            self.var_to_dvar = self.var_to_dvar | fm.var_to_dvar

            
            self.cache_vars_list = fm.cache_vars_list
            self.type_cache_size = fm.type_cache_size
            self.type_to_stack_and_ptr_names = fm.type_to_stack_and_ptr_names
            # self.while_stack = fm.while_stack
            tmp_declares = fm.tmp_declares
            
            print("Ho-oH")
            print(tmp_declares)
            
            
            for t, exprs in fm.cache_vars_list.items():
                print(f'type: {t} | exprs: {exprs}')
                t_str = type_to_string(t)
                stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                
                # will not work if variables asigned outside of while but he doesn't test
                
                if fm.while_stack:
                   tmp_declares = [loma_ir.Declare(stack_name, loma_ir.Array(t, fm.while_stack)), loma_ir.Declare(stack_ptr_name, loma_ir.Int(), loma_ir.ConstInt(0))] + tmp_declares
                else:
                    tmp_declares.append(loma_ir.Declare(stack_name, loma_ir.Array(t, len(exprs))))
                    tmp_declares.append(loma_ir.Declare(stack_ptr_name, loma_ir.Int(), loma_ir.ConstInt(0)))
            mutated_forward = tmp_declares + mutated_forward
            
            print()
            print(mutated_forward)
            print()
            print("printing temp_dec")
            
            print(tmp_declares)
            # Reverse pass
            self.adj_count = 0
            self.in_assign = False
            self.adj_declaration = []
            reversed_body = [self.mutate_stmt(stmt)
                             for stmt in reversed(node.body)]
            reversed_body = irmutator.flatten(reversed_body)
            
            
            return loma_ir.FunctionDef(
                diff_func_id,
                new_args,
                mutated_forward + self.adj_declaration + reversed_body,
                node.is_simd,
                ret_type=None,
                lineno=node.lineno)

        def mutate_return(self, node):
            debug_node(node)
            self.backward_out = None
            if isinstance(node.val, loma_ir.Call):
                    print("dialga")
                    # print(self.)
                    self.backward_out = (self.return_var_id, loma_ir.Float())
            # self.backward_out = (self.var_to_dvar[call_arg.id], call_arg.t)
            # Propagate to each variable used in node.val
            self.adj = loma_ir.Var(self.return_var_id, t=node.val.t)
            
            out = self.mutate_expr(node.val)
            self.backward_out = None
            return out

        def mutate_declare(self, node):
            if node.val is not None:
                if node.t == loma_ir.Int():
                    return []

                self.adj = loma_ir.Var(self.var_to_dvar[node.target])

                self.backward_out = None

                if isinstance(node.val, loma_ir.Call):
                    self.backward_out = (self.var_to_dvar[node.target], node.t)

                new_node = self.mutate_expr(node.val)

                self.backward_out = None

                return new_node
            else:
                return []

        def mutate_assign(self, node):
            if node.val.t == loma_ir.Int():
                stmts = []
                # restore the previous value of this assignment
                t_str = type_to_string(node.val.t)
                _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                stmts.append(loma_ir.Assign(stack_ptr_var,
                                            loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                cache_var_expr, cache_target = self.cache_vars_list[node.val.t].pop(
                )
                stmts.append(loma_ir.Assign(cache_target, cache_var_expr))
                return stmts

            self.adj = var_to_differential(node.target, self.var_to_dvar)
            if check_lhs_is_output_arg(node.target, self.output_args):
                # if the lhs is an output argument, then we can safely
                # treat this statement the same as "declare"

                print(node.val)
                return self.mutate_expr(node.val)
            else:
                stmts = []
                # restore the previous value of this assignment
                t_str = type_to_string(node.val.t)
                _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                stmts.append(loma_ir.Assign(stack_ptr_var,
                                            loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                cache_var_expr, cache_target = self.cache_vars_list[node.val.t].pop(
                )
                stmts.append(loma_ir.Assign(cache_target, cache_var_expr))

                # First pass: accumulate
                self.in_assign = True
                self.adj_accum_stmts = []

                self.backward_out = None
                if isinstance(node.val, loma_ir.Call):
                    self.backward_out = (
                        self.var_to_dvar[node.target.id], node.target.t)

                stmts += self.mutate_expr(node.val)
                self.backward_out = None
                self.in_assign = False

                # zero the target differential
                stmts += assign_zero(var_to_differential(node.target,
                                     self.var_to_dvar))

                # Accumulate the adjoints back to the target locations
                stmts += self.adj_accum_stmts
                return stmts

        def mutate_ifelse(self, node):
            # HW3: TODO

            new_else_stmts = irmutator.flatten(
                [self.mutate_stmt(stmt) for stmt in reversed(node.else_stmts)])
            new_then_stmts = irmutator.flatten(
                [self.mutate_stmt(stmt) for stmt in reversed(node.then_stmts)])

            return loma_ir.IfElse(node.cond, new_then_stmts, new_else_stmts)
        
                
        def mutate_call_stmt(self, node):
            # HW3: TODO
            if node.call.id == "atomic_add":
                
                # old_adj = self.adj
                self.adj = loma_ir.Var(self.var_to_dvar[node.call.args[0].id]) 
                print(f"second arg: {self.mutate_expr(node.call.args[1])}")
                
                # self.adj = old_adj
                return self.mutate_expr(node.call.args[1])
            
            print("YARE YARE")
            debug_node(node)
            self.in_call_stmt = True
            self.backward_out = None
            func_def = funcs[node.call.id]
            
            for call_arg, func_arg in zip(node.call.args, func_def.args):
                if isinstance(func_arg.i, loma_ir.In): 
                    print("Lego")
                    continue
                if call_arg.t == loma_ir.Int():
                    stmts = []
                    # restore the previous value of this assignment
                    t_str = type_to_string(call_arg.t)
                    _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                    stack_ptr_var = loma_ir.Var(
                        stack_ptr_name, t=loma_ir.Int())
                    stmts.append(loma_ir.Assign(stack_ptr_var,
                                                loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                    cache_var_expr, cache_target = self.cache_vars_list[call_arg.t].pop(
                    )
                    stmts.append(loma_ir.Assign(cache_target, cache_var_expr))
                    return stmts
                
                print("Hello")
                if check_lhs_is_output_arg(call_arg, self.output_args): 
                    return self.mutate_call(node.call)
                

                self.adj = var_to_differential(call_arg, self.var_to_dvar)


                stmts = []
                # restore the previous value of this assignment
                t_str = type_to_string(call_arg.t)
                
                
                _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                stmts.append(loma_ir.Assign(stack_ptr_var, loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                cache_var_expr, cache_target = self.cache_vars_list[call_arg.t].pop()
                
                stmts.append(loma_ir.Assign(cache_target, cache_var_expr))

                # First pass: accumulate
                self.adj_accum_stmts = []

                # self.backward_out = (self.var_to_dvar[call_arg.id], call_arg.t)
                
                
                stmts += self.mutate_call(node.call)
                
                self.backward_out = None
                self.in_assign = False
                
                # zero the target differential
                stmts += assign_zero(var_to_differential(call_arg, self.var_to_dvar))
                
                
                # Accumulate the adjoints back to the target locations
                stmts += self.adj_accum_stmts
                return stmts
            
            out = self.mutate_call(node.call)
            self.in_call_stmt = False
            self.backward_out = None
            return [out]
            
            return super().mutate_call_stmt(node)

        def mutate_while(self, node):
            # HW3: TODO
            out = self.mutate_while_backward(node, ctr_id=0)
            
            print("exit")
            return out
            return super().mutate_while(node)
        
        def mutate_while_backward(self, node, ctr_id=0):
            # ctr_name = f"ctr{ctr_id}"
            # cond = loma_ir.BinaryOp(loma_ir.Greater(), ctr_name, loma_ir.ConstInt(0))
            # while_node = loma_ir.While(cond, max_iter=node.max_iter, )
            
            
            
            ctr_name, ctr = f"ctr{ctr_id}", None
            ctr_ptr_name, ctr_stack_ptr = None, None
            
            if ctr_id > 0:
                # ctr = loma_ir.Declare(ctr_name, loma_ir.Array(loma_ir.Int(), self.while_stack))
                
                ctr_ptr_name = f"{ctr_name}_ptr"
                # ctr_stack_ptr = loma_ir.Declare(ctr_ptr_name, loma_ir.Int(), loma_ir.ConstInt(0))

            # self.while_stack *= node.max_iter
                
            # self.tmp_declares.append(ctr)
            
            # if ctr_stack_ptr: self.tmp_declares.append(ctr_stack_ptr)
            
            
            body = []
            for stmt in reversed(node.body):
                if isinstance(stmt, loma_ir.While):
                    ctr = loma_ir.Var(f"ctr{ctr_id+1}_ptr")
                    
                    dec_stack_ptr = loma_ir.Assign(ctr, loma_ir.BinaryOp(loma_ir.Sub(), ctr, loma_ir.ConstInt(1)))
                    print(f"dec_stack_ptr: {dec_stack_ptr}")
                    body.append(dec_stack_ptr)

                    body += self.mutate_while_backward(stmt, ctr_id + 1) 
                else: body += self.mutate_stmt(stmt)
            # out = []
            
            new_cond = None
            if ctr_id > 0: 
                index_location = loma_ir.ArrayAccess(loma_ir.Var(ctr_name), loma_ir.Var(ctr_ptr_name))
                body.append(loma_ir.Assign(index_location, loma_ir.BinaryOp(loma_ir.Sub(), index_location, loma_ir.ConstInt(1))))

                ctr_access = loma_ir.ArrayAccess(loma_ir.Var(ctr_name), loma_ir.Var(ctr_ptr_name))
                new_cond = loma_ir.BinaryOp(loma_ir.Greater(), ctr_access, loma_ir.ConstInt(0))
                
                # out.append(loma_ir.Assign(loma_ir.Var(ctr_ptr_name), loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var(ctr_ptr_name), loma_ir.ConstInt(1))))
            else:
                body.append(loma_ir.Assign(loma_ir.Var(ctr_name), loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.Var(ctr_name), loma_ir.ConstInt(1))))
                new_cond = loma_ir.BinaryOp(loma_ir.Greater(), loma_ir.Var(ctr_name), loma_ir.ConstInt(0))
                
            while_node = loma_ir.While(new_cond, node.max_iter, irmutator.flatten(body))
            
            return [while_node]
            # return out
            
            
        def mutate_var(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                                target_expr, overwrite=False)
                return [accum_deriv(target_expr, self.adj, overwrite=True)]
            else:
                return [accum_deriv(var_to_differential(node, self.var_to_dvar),
                                    self.adj, overwrite=False)]

        def mutate_const_float(self, node):
            return []

        def mutate_const_int(self, node):
            return []

        def mutate_array_access(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                                target_expr, overwrite=False)
                return [accum_deriv(target_expr, self.adj, overwrite=True)]
            else:
                return [accum_deriv(var_to_differential(node, self.var_to_dvar),
                                    self.adj, overwrite=False)]

        def mutate_struct_access(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                                target_expr, overwrite=False)
                return [accum_deriv(target_expr, self.adj, overwrite=True)]
            else:
                return [accum_deriv(var_to_differential(node, self.var_to_dvar),
                                    self.adj, overwrite=False)]

        def mutate_add(self, node):
            left = self.mutate_expr(node.left)
            right = self.mutate_expr(node.right)
            return left + right

        def mutate_sub(self, node):
            old_adj = self.adj
            left = self.mutate_expr(node.left)
            self.adj = loma_ir.BinaryOp(loma_ir.Sub(),
                                        loma_ir.ConstFloat(0.0), old_adj)
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_mul(self, node):
            # z = x * y
            # dz/dx = dz * y
            # dz/dy = dz * x
            old_adj = self.adj
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                                        node.right, old_adj)
            left = self.mutate_expr(node.left)
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                                        node.left, old_adj)
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_div(self, node):
            # z = x / y
            # dz/dx = dz / y
            # dz/dy = - dz * x / y^2
            old_adj = self.adj
            self.adj = loma_ir.BinaryOp(loma_ir.Div(),
                                        old_adj, node.right)
            left = self.mutate_expr(node.left)
            # - dz
            self.adj = loma_ir.BinaryOp(loma_ir.Sub(),
                                        loma_ir.ConstFloat(0.0), old_adj)
            # - dz * x
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                                        self.adj, node.left)
            # - dz * x / y^2
            self.adj = loma_ir.BinaryOp(loma_ir.Div(),
                                        self.adj, loma_ir.BinaryOp(loma_ir.Mul(), node.right, node.right))
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_call(self, node):
            match node.id:
                case 'sin':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.Call(
                            'cos',
                            node.args,
                            lineno=node.lineno,
                            t=node.t),
                        old_adj,
                        lineno=node.lineno)
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'cos':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        loma_ir.ConstFloat(0.0),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            loma_ir.Call(
                                'sin',
                                node.args,
                                lineno=node.lineno,
                                t=node.t),
                            self.adj,
                            lineno=node.lineno),
                        lineno=node.lineno)
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'sqrt':
                    assert len(node.args) == 1
                    # y = sqrt(x)
                    # dx = (1/2) * dy / y
                    old_adj = self.adj
                    sqrt = loma_ir.Call(
                        'sqrt',
                        node.args,
                        lineno=node.lineno,
                        t=node.t)
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.ConstFloat(0.5), self.adj,
                        lineno=node.lineno)
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        self.adj, sqrt,
                        lineno=node.lineno)
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'pow':
                    assert len(node.args) == 2
                    # y = pow(x0, x1)
                    # dx0 = dy * x1 * pow(x0, x1 - 1)
                    # dx1 = dy * pow(x0, x1) * log(x0)
                    base_expr = node.args[0]
                    exp_expr = node.args[1]

                    old_adj = self.adj
                    # base term
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        self.adj, exp_expr,
                        lineno=node.lineno)
                    exp_minus_1 = loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        exp_expr, loma_ir.ConstFloat(1.0),
                        lineno=node.lineno)
                    pow_exp_minus_1 = loma_ir.Call(
                        'pow',
                        [base_expr, exp_minus_1],
                        lineno=node.lineno,
                        t=node.t)
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        self.adj, pow_exp_minus_1,
                        lineno=node.lineno)
                    base_stmts = self.mutate_expr(base_expr)
                    self.adj = old_adj

                    # exp term
                    pow_expr = loma_ir.Call(
                        'pow',
                        [base_expr, exp_expr],
                        lineno=node.lineno,
                        t=node.t)
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        self.adj, pow_expr,
                        lineno=node.lineno)
                    log = loma_ir.Call(
                        'log',
                        [base_expr],
                        lineno=node.lineno)
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        self.adj, log,
                        lineno=node.lineno)
                    exp_stmts = self.mutate_expr(exp_expr)
                    self.adj = old_adj
                    return base_stmts + exp_stmts
                case 'exp':
                    assert len(node.args) == 1
                    exp = loma_ir.Call(
                        'exp',
                        node.args,
                        lineno=node.lineno,
                        t=node.t)
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        self.adj, exp,
                        lineno=node.lineno)
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'log':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        self.adj, node.args[0])
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'int2float':
                    # don't propagate the derivatives
                    return []
                case 'float2int':
                    # don't propagate the derivatives
                    return []
                case 'atomic_add':
                    print(node)
                case _:
                    # HW3: TODO

                    func_def = funcs[node.id]
                    diff_func_id = func_to_rev[node.id]
                    d_args = []
                    
                    print(f"checking {node.id}")
                    print(self.backward_out)
                    print(node.args)
                    print(func_def.args)
                    # print(diff_func_id)
                    # print(func_def)
                    for call_arg, func_def_arg in zip(node.args, func_def.args):
                        # print(call_arg, func_def_arg)
                        d_id = self.var_to_dvar[call_arg.id]
                        if isinstance(func_def_arg.i, loma_ir.In):
                            d_args.append(call_arg)
                        else:
                            pass
                        d_arg = loma_ir.Var(d_id, t=call_arg.t)
                        d_args.append(d_arg)
                        
                    debug_node(node)
                    if self.backward_out:
                        id, t = self.backward_out
                        d_args.append(loma_ir.Var(id=id, t=t))

                    print(f"d_args: {d_args}")
                    call = loma_ir.Call(diff_func_id, d_args)
                    # return []
                    return [loma_ir.CallStmt(call=call)]
                    assert False

    return RevDiffMutator().mutate_function_def(func)
