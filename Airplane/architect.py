NUM_LEVELS = 6
ADDEND = 6
NUM_CONV_PER_LEVEL = 2

lowest_dim = 70
highest_dim = lowest_dim/2

for i in range(NUM_LEVELS):
    highest_dim = highest_dim*2 + NUM_CONV_PER_LEVEL*ADDEND
    print(str.format('{0} - {1} - {2}',highest_dim,highest_dim - ADDEND,highest_dim - 2*ADDEND))

print("#############################################")

for i in range(NUM_LEVELS):
    lowest_dim = lowest_dim*2 + NUM_CONV_PER_LEVEL*ADDEND
    print(str.format('{0} - {1} - {2}',lowest_dim - 2*ADDEND,lowest_dim - ADDEND,lowest_dim))

# print(highest_dim)
