import json

# Read data from 'data.json' file
# with open('/fs/nexus-scratch/sjxu/svd-temporal-controlnet/relighting/training.json', 'r') as infile:
#     data_list = json.load(infile)

data_list = [
    json.loads(line) for line in open(f"/fs/nexus-scratch/sjxu/svd-temporal-controlnet/relighting/training.json", "r").read().splitlines()
]

input_file = "/fs/nexus-scratch/sjxu/svd-temporal-controlnet/relighting/training.json"
output_file = "/fs/nexus-scratch/sjxu/svd-temporal-controlnet/relighting/training_edit.json"

# Values to filter out
filter_values = {"D", "E", "F"}

# # Filter out entries with target_image or conditioning_image in filter_values
# filtered_data = [
#     item for item in data_list
#     if item["target_image"] not in filter_values and item["conditioning_image"] not in filter_values
# ]

# print(filtered_data)
# # Save the filtered data to a new JSON file
# with open('/fs/nexus-scratch/sjxu/svd-temporal-controlnet/relighting/training_edit.json', 'w') as outfile:
#     json.dump(filtered_data, outfile, indent=4)

# print("Filtered data saved to 'filtered_data.json'")

# Read the file and filter the data
filtered_data = []
with open(input_file, 'r') as infile:
    for line in infile:
        item = json.loads(line)
        if item["target_image"] not in filter_values and item["conditioning_image"] not in filter_values:
            filtered_data.append(item)

# Write the filtered data to the output file
with open(output_file, 'w') as outfile:
    for item in filtered_data:
        json_line = json.dumps(item)
        outfile.write(json_line + '\n')

print(f"Filtered data saved to '{output_file}'")