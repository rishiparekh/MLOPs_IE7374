

def add(a:float|int,b:float|int) -> float:
    return float(a+b)

def sub(a:float|int,b:float|int) -> float:
    return float(a-b)

def prod(a:float|int,b:float|int) -> float:
    return float(a*b)

def addThree(a:float|int, b:float|int,) -> float:
    #Adding output from the three functions above
    return (add(a,b) + sub(a,b) + prod(a,b) )



