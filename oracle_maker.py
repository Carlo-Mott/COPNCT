from collections import defaultdict
def replace_and_cumulative_sum(lst):
    # Define the replacement mapping
    mapping = {
        1: 3000,
        2: 2994,
        3: 3002,
        4: 3005,
        5: 2998,
        6: 3003,
        7: 2994,
        8: 3004,
        9: 3007,
        10: 2994,
        11: 3002,
        12: 1996,
        13: 4008,
        14: 3001,
        15: 3003,
        16: 3002,
        17: 2990,
        18: 3006,
        19: 2993
    }


    # Replace each number in the list according to the mapping
    mapped_lst = [mapping[num] for num in lst]

    # Compute the cumulative sum
    cumulative_sum = []
    current_sum = 0
    for num in mapped_lst:
        current_sum += num
        cumulative_sum.append(current_sum)

    return cumulative_sum[:-1] #the last index would be the end of the dataset, not relevant

def replace_with_occurrences(lst):
    count_dict = defaultdict(int)  # to track occurrences of each number
    result = []
    
    for num in lst:
        count_dict[num] += 1
        result.append(count_dict[num])
    
    return replace_and_cumulative_sum(result)

# Example usage:
if __name__ == "__main__":
    input_list = [1, 2, 1, 3, 2, 1]
    output = replace_with_occurrences(input_list)
    print("Input:", input_list)
    print("Output:", output)

