import logging

# 配置日志记录器
logging.basicConfig(filename='example.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')

def add(a, b):
    """
    Returns the sum of two numbers.
    """
    logging.debug("Adding %s and %s", a, b)
    result = a + b
    logging.debug("Result: %s", result)
    return result

def multiply(a, b):
    """
    Returns the product of two numbers.
    """
    logging.debug("Multiplying %s and %s", a, b)
    result = a * b
    logging.debug("Result: %s", result)
    return result

# 使用函数记录日志
x = 2
y = 3
logging.info("Starting calculation with %s and %s", x, y)
result1 = add(x, y)
result2 = multiply(x, y)
logging.info("Calculation finished: %s %s", result1, result2)