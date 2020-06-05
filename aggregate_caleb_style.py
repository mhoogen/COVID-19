from data_processing.process_data import createProcessedFiles
from util_ import in_out as io

c = createProcessedFiles()

[attr, values] = c.read_files_and_calculate_attributes('../data/fake_data.csv', '../result.csv', 0)
print(attr)
print(values)
