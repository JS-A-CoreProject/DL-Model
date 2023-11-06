import random
import string

def random_string():
    char = string.ascii_letters + string.digits
    
    ran_str = ''.join(random.choice(char) for _ in range(16))
    
    return ran_str