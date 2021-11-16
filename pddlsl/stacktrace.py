#!/usr/bin/env python3

import os.path
import sys, traceback, types, linecache
from colors import red, blue, yellow, green

# note: arraytypes is no longer used, but it is kept for compatibility.

def print_object(o, include_private=False, threshold=3, include_self=True, ignore={}, ignore_type={}, arraytypes={}):
    maxlinelen=1000
    maxlen=20
    ignore_type = set(ignore_type)
    ignore = set(ignore)
    def err(thing):
        print(thing, file=sys.stderr)

    def get(o,key):
        if isinstance(o, dict):
            return o[key]
        else:
            return getattr(o, key)

    ignore_type |= {types.FunctionType, types.ModuleType, type}
    def include(o,key):
        return (include_private or ("__" not in key))       \
          and key not in ignore                             \
          and not isinstance(get(o,key),tuple(ignore_type))

    def remove_array(thing):
        available_arraylike_fields = [
            name
            for name in ["dtype","shape","device","layout"]
            if hasattr(thing, name)
        ]
        if len(available_arraylike_fields) > 0:
            return "<"+" ".join([
                type(thing).__name__,
                *[ str(getattr(thing,name)) for name in available_arraylike_fields ]])+">"
        else:
            return thing

    def printer(thing):
        if isinstance(thing,list):
            if len(thing) > threshold:
                return [printer(remove_array(o)) for o, _ in [*list(zip(thing, range(threshold))),(f"...<{len(thing)-threshold} more>",None)]]
            else:
                return [printer(remove_array(o)) for o in thing]
        elif isinstance(thing,tuple):
            if len(thing) > threshold:
                return tuple([printer(remove_array(o)) for o in [*list(zip(thing, range(threshold))),(f"...<{len(thing)-threshold} more>",None)]])
            else:
                return tuple([printer(remove_array(o)) for o in thing])
        elif isinstance(thing,dict):
            return {k:printer(remove_array(v)) for k,v in thing.items()}
        elif isinstance(thing,str):
            return thing[:500]
        elif isinstance(thing,bytes):
            return thing[:500]
        else:
            return remove_array(thing)

    def multi_indent(width,string):
        lines = [ line[:maxlinelen] for line in string.splitlines() ]
        return ("\n"+" "*width).join(lines)

    try:
        zip(o)
    except TypeError:
        print(o)
        return

    for key in o:
        try:
            if include(o,key):
                maxlen = max(maxlen,len(key))
                if include_self and (key == "self"):
                    __self = get(o,key)
                    for key in vars(__self):
                        if include(__self,key):
                            maxlen = max(maxlen,len(key)+5) # + 5 for "self."
        except Exception as e:
            print(e)
            pass

    maxlen += 10                # buffer

    for key in o:
        if include(o,key):
            try:
                err("{} = {}".format(yellow(str(key)).rjust(maxlen+4), # +4 for ANSI color
                                     multi_indent(maxlen-2,repr(printer(get(o,key)))))) # +4+3 for " = "
            except Exception as e:
                err("{} = Error printing {} : {}".format(red(str(key)).rjust(maxlen+4),
                                                         type(get(o,key)),
                                                         e))
            if include_self and (key == "self"):
                __self = get(o,key)
                for key in vars(__self):
                    if include(__self,key):
                        try:
                            err("{} = {}".format((green("self")+"."+yellow(str(key))).rjust(maxlen+13), # 5+2*4 for double ANSI color
                                                 multi_indent(maxlen-2,repr(printer(get(__self,key))))))
                        except Exception as e:
                            err("{} = Error printing {} : {}".format(red("self."+str(key)).rjust(maxlen+4),
                                                                     type(get(__self,key)),
                                                                     e))


def format(exit=True,threshold=3,include_self=True,ignore={},ignore_type={},include_private=False,arraytypes={}):
    print("Fancy Traceback (most recent call last):",file=sys.stderr)
    type, value, tb = sys.exc_info()

    for f, f_lineno in traceback.walk_tb(tb):
        co = f.f_code
        f_filename = co.co_filename
        f_name = co.co_name
        linecache.lazycache(f_filename, f.f_globals)
        f_locals = f.f_locals
        f_line = linecache.getline(f_filename, f_lineno).strip()

        print(" ",green("File"),os.path.relpath(f_filename),green("line"),f_lineno,green("function"),f_name,":",f_line,file=sys.stderr)
        print_object(f_locals,threshold=threshold,include_self=include_self,ignore=ignore,ignore_type=ignore_type,include_private=include_private,arraytypes=arraytypes)
        print(file=sys.stderr)


    print(file=sys.stderr)
    print(*(traceback.format_exception_only(type,value)),file=sys.stderr)
    if exit:
        sys.exit(1)

def fn1():
    a = 1
    b = 0
    fn2(a,b)

def fn2(a,b):
    return a/b


class MultiLineString:
    def __repr__(self):
        return "aaa\nbbb\nccc"

class A():
    def __init__(self):
        self.x = 10
        self.y = B()
        self.field_name = 10
        # self.super_long_field_name = 10
        # self.gigantic_field_name = 10
        # self.long_gigantic_field_name = 10
        # self.stupidly_long_gigantic_field_name = 10
        # self.duper_stupidly_long_gigantic_field_name = 10
        self.super_duper_stupidly_long_gigantic_field_name = 10
        self.multi_line_string = MultiLineString()
    def m(self,y,
          super_duper_long_variable,
          *args):
        z = B()
        multi_line_string = MultiLineString()
        1/0

class B():
    def __str__(self):
        1/0
    def __repr__(self):
        1/0

if __name__ == '__main__':
    try:
        fn1()
    except Exception:
        print("## fancy stack trace ########################################",file=sys.stderr)
        format(exit=False)
        print("## standard stack trace ########################################",file=sys.stderr)
        traceback.print_exc()

    try:
        a = A()
        a.m(3,4)
    except Exception:
        print("## fancy stack trace ########################################",file=sys.stderr)
        format(exit=False)
        print("## standard stack trace ########################################",file=sys.stderr)
        traceback.print_exc()
