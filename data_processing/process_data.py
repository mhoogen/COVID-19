import numpy as np
from util_ import util
from util_ import in_out as io
import scipy.stats as sc
import time
import math
import copy
import datetime

class createProcessedFiles:


    def __init__(self):
        self.writer = ''
        self.patient_dict = {}     #Set (no duplicate values)
        self.headers = []          #list
        self.number_of_days_considered = 1 # First 1 day, should be 6 days (cf. Caleb)
        self.day_change = 23 # 11pm is the time at which the day changes.


        # The numerical features present in the current dataset.
        self.numerical_features = ["GCS","WEIGHT","ADMITWT","NBPSYS","NBPDIAS","NBPMEAN",
                                   "NBP","SBP","DBP","MAP","HR","RESP","SPO2","CVP",
                                   "PAPMEAN","PAPSD","CRDINDX","SVR","COTD","COFCK","PCWP",
                                   "PVR","NA","K","CL","CO2","GLUCOSE","BUN","CREATININE",
                                   "MG","AST","ALT","CA","IONCA","TBILI","DBILI","TPROTEIN",
                                   "ALBUMIN","LACTATE","TROPONIN","HCT","HG","PLATELETS","INR",
                                   "PT","PTT","WBC","RBC","TEMP","ARTBE","ARTCO2","ARTPACO2",
                                   "ARTPAO2","ARTPH","FIO2SET","PEEPSET","RESPTOT","RESPSET",
                                   "RESPSPON","PIP","PLATEAUPRES","TIDVOLOBS","TIDVOLSET","TIDVOLSPON",
                                   "SAO2", "hrmhb", "hrmsa", "hrmva", "orientation_ord", "riker_sas_ord",
                                   "vent_mode_ord", "iabp_ord"]
        self.numerical_features = [feat.lower() for feat in self.numerical_features]

        self.features_to_be_excluded = ["ICUSTAY_EXPIRE_FLG","DAYSFROMDISCHTODEATH","AGEATDEATH","ICUSTAY_INTIME",
                                        "ICUSTAY_OUTTIME","ICUSTAY_LOS","ICUSTAY_FIRST_CAREUNIT","ICUSTAY_LAST_CAREUNIT", "TIME"]
        self.features_to_be_excluded = [feat.lower() for feat in self.features_to_be_excluded]

        self.fixed_features = ['ID', 'label', 'sex_bin', 'white']
        self.fixed_features = [feat.lower() for feat in self.fixed_features]

        self.slope_windows = {}
        self.four_twenty_eight_windows = ["NBPSYS","NBPDIAS","NBPMEAN",
                                   "NBP","SBP","DBP","MAP","HR","RESP","SPO2","CVP",
                                   "PAPMEAN","PAPSD","CRDINDX","SVR","COTD","COFCK","PCWP",
                                   "PVR"]
        self.four_twenty_eight_windows = [feat.lower() for feat in self.four_twenty_eight_windows]

        for attribute in self.four_twenty_eight_windows:
            self.slope_windows[attribute] = [4,28]

        self.twenty_eight_windows = ["GCS","WEIGHT","NA","K","CL","CO2","GLUCOSE","BUN","CREATININE",
                                   "MG","AST","ALT","CA","IONCA","TBILI","DBILI","TPROTEIN",
                                   "ALBUMIN","LACTATE","TROPONIN","HCT","HG","PLATELETS","INR",
                                   "PT","PTT","WBC","RBC","TEMP","ARTBE","ARTCO2","ARTPACO2",
                                   "ARTPAO2","ARTPH","FIO2SET","PEEPSET","RESPTOT","RESPSET",
                                   "RESPSPON","PIP","PLATEAUPRES","TIDVOLOBS","TIDVOLSET","TIDVOLSPON",
                                   "SAO2"]
        self.twenty_eight_windows = [feat.lower() for feat in self.twenty_eight_windows]

        for attribute in self.twenty_eight_windows:
            self.slope_windows[attribute] = [28]

    # Identify IDs (combi of patient ID and stay ID)
    def determine_complete_patient_ids(self,File):
        rows = io.read_csv(File, ',')
        current_ID  = 0
        previous_ID = 0
        complete_patient_ID = []
        counter = 0

        for row in rows:
            if (counter > 0):
                current_ID  = str(row[0] + row[1])
                if not current_ID in complete_patient_ID:
                    complete_patient_ID.append(current_ID)
            counter += 1
        return  complete_patient_ID


    def perform_regression(self, times, values):
        new_values = []
        if (len(values)) < 2:
            return 0

        #slope, intercept, r_value, p_value, std_err = sc.linregress(times, values)
        X = np.array(np.column_stack(times))
        X = np.vstack([np.column_stack(times), np.ones(len(times))]).T
        Y = np.array(values)
        W = np.linalg.lstsq(X,Y)[0]
        return W[0]

    # For windows of less than 24 hours.
    def derive_windows(self, window):
        window_number = 1
        windows_dict = {}
        start_hours = self.day_change
        end_hours = self.day_change

        while math.floor(float(end_hours-self.day_change)/24) <= self.number_of_days_considered:
            windows_dict[window_number] = {'start_hour':0, 'end_hour':0}
            windows_dict[window_number]['start_hour'] = start_hours
            end_hours = start_hours + window
            windows_dict[window_number]['end_hour'] = end_hours
            if window > 24:
                start_hours = end_hours - (window-24)
            else:
                start_hours = end_hours
            window_number += 1
        return windows_dict

    def derive_values_for_window(self, attribute, timestamps, values, window, type):
        windows_dict = self.derive_windows(window)
        new_values = []
        new_attributes = []

        if type == 0:
            expected_number_frames = int(math.ceil(float(self.number_of_days_considered * 24) / window))
        elif type == 1:
            expected_number_frames = int(math.ceil(float(24) / window))
        # First transform the time stamps to hours and deduct the initial timestamp.

        if len(timestamps) > 0:
            start_hour = time.localtime(timestamps[0]).tm_hour
            if start_hour < self.day_change:
                start_hour += 24
            hourly_timestamps = [((float(ts)-float(timestamps[0]))/(60*60)) + start_hour for ts in timestamps]
            minute_timestamps = [float(ts)/60 for ts in timestamps]
        else:
            hourly_timestamps = []
            minute_timestamps = []

        for frame in range(1, expected_number_frames+1):
            start_hours = windows_dict[frame]['start_hour']
            end_hours = windows_dict[frame]['end_hour']
            # Get all time points within this range

            indices = [i for i,j in enumerate(hourly_timestamps) if j < end_hours and j >= start_hours]
            new_attributes.append('slope_windows_' + str(frame*window) + '_' + str(attribute))

            if len(indices) > 0:
                new_values.append(self.perform_regression([minute_timestamps[i] for i in indices], [values[i] for i in indices]))
            else:
                new_values.append(0)

        return new_attributes, new_values


    # For this category we want to find the
    def derive_numerical_features(self, attribute, timestamps, values, type):
        # Select the timepoints and values that have a value
        remove_indices = [i for i,val in enumerate(values) if val == '']
        values = [float(i) for j, i in enumerate(values) if j not in remove_indices]
        timestamps = [float(i) for j, i in enumerate(timestamps) if j not in remove_indices]

        new_attributes = []
        new_values = []
        if self.slope_windows.has_key(attribute):
            windows = self.slope_windows[attribute]
        else:
            windows = [6] # 6 is considered to be the default value.

        for window in windows:
            window_attributes, window_values = self.derive_values_for_window(attribute, timestamps, values, window, type)
            new_attributes.extend(window_attributes)
            new_values.extend(window_values)

        # And also calculate the summary statistics:
        new_attributes.append('mean_' + attribute)
        new_attributes.append('max_' + attribute)
        new_attributes.append('min_' + attribute)
        new_attributes.append('std_' + attribute)
        if len(values) > 0:
            new_values.append(np.mean(values))
            new_values.append(max(values))
            new_values.append(min(values))
            new_values.append(np.std(values))
        else:
            new_values.extend([0,0,0,0])


        return new_attributes, new_values

    def determine_type(self, list):
        list_type = 'numeric'
        for element in list:
            if not type(element) is float and not type(element) is int:
                list_type = 'non numeric'
        return list_type

    # The rest category, assumed for now that we simply use the mean value.
    def derive_simple_aggregate(self, attribute, timestamps, values):
        # First get the actual non null values
        final_value = 0
        filtered_values = [v for v in values if v != '']
        if (len(filtered_values) > 0):
            # Determine the type: for numerical we take the average, for
            # categorial we use the most frequently occurring item.
            if self.determine_type(filtered_values) == 'numeric':
                final_value = sum(filtered_values)/len(values)
            else:
                final_value = max(values)
                try:
                    final_value = float(final_value)
                except ValueError:
                    final_value = 0
        return [attribute], [final_value]


    def aggregate_attribute(self, attribute, timestamps, values, type):
        if attribute == 'id' or attribute == 'label':
            return [attribute], [values[0]]
        elif attribute in self.features_to_be_excluded:
            return [], []
        elif attribute in self.numerical_features:
            return self.derive_numerical_features(attribute, timestamps, values, type)
        else:
            # Binary features assumed for the remainder.
            return self.derive_simple_aggregate(attribute, timestamps, values)

    def aggregate_day_sdas_das(self, attributes, patient_measurements, day_count, type):

        final_attributes = []
        final_values = []
        timestamps = patient_measurements[:,attributes.index('time')].tolist()
        for i in range(0, len(attributes)):
            [attributes_found, values_found] = self.aggregate_attribute(attributes[i], timestamps, patient_measurements[:,i].tolist(), type)
            if len(attributes) > 0:
                final_attributes.extend(attributes_found)
                if i > 0:
                    final_values.extend(map(float, values_found))
                else:
                    final_values.extend(values_found)


        final_attributes.insert(1, 'day')
        final_values.insert(1, day_count)
        return final_attributes, np.array(final_values)

    def to_struct_time(self, timestamps):
        struct_timestamps = []
        for timestamp in timestamps:
            struct_timestamps.append(time.localtime(float(timestamp)))
        return struct_timestamps

    def time_point_present(self, current_hours, current_minutes, timestamps):
        for i in range(0, len(timestamps)):
            if timestamps[i].tm_hour == current_hours and timestamps[i].tm_min == current_minutes:
                return i
        return -1

    def aggregate_day(self, attributes, patient_measurements, day_count, type):
        [attr, values] = self.aggregate_day_sdas_das(attributes, patient_measurements, day_count, type)
        return [attr, values.astype(np.float64, copy=False)]

    # Aggregation for both the SDAS and DAS approach, if more fine-grained predictions are
    # required this should slightly change.

    def aggregate(self, type):

        # The numpy data structure in which all aggregated data will be stored, the day will replace the time
        # The ID will also be included.
        aggr_set = np.zeros((0, 0))
        new_attributes = []
        count = 0
        row_count = 0

        for ID in self.patient_dict:
            if (count % 1000 == 0):
                print('====== ' + str(count))
            count += 1
            # we start with day 1 and continue until either
            day_count = 1
            index = 0
            daily_set = np.zeros((0, len(self.headers)+1))
            prev_time_point = self.patient_dict[ID]['time'][index]
            extended_headers = ['id']
            extended_headers.extend(self.headers)

            while day_count <= self.number_of_days_considered and index < len(self.patient_dict[ID]['time']):
                curr_time_point = self.patient_dict[ID]['time'][index]

                if ((prev_time_point.tm_hour < self.day_change and curr_time_point.tm_hour >= self.day_change) or
                     (prev_time_point.tm_mday != curr_time_point.tm_mday and
                      (prev_time_point.tm_hour < self.day_change and curr_time_point.tm_hour < self.day_change))):
                    # We have changed to a new day, aggregate the current dataset and add it to the full set.
                    [attr, aggregated_values] = self.aggregate_day(extended_headers, daily_set, day_count, type)

                     # Set the attributes in case we haven't done so yet.
                    if len(new_attributes) == 0:
                        new_attributes = attr
                        self.writer.writerow(attr)
                        aggr_set = np.zeros((0, len(attr)))
                    else:
                        aggr_set = np.append(aggr_set, np.column_stack(aggregated_values), axis=0)
                    if type == 1:
                        # For the case of the DAS model we start with an empty set again
                        daily_set = np.zeros((0, len(self.headers)+1))
                    day_count += 1
                    if day_count > self.number_of_days_considered:
                        break

                # Look at the current data entry
                current_values = []
                current_values.append(ID)
                # Determine the values of the current case
                for i in range(0, len(self.headers)):
                    if self.headers[i] == 'time':
                        #edit by Ali (for Windows)
                        current_values.append(((datetime.datetime(*self.patient_dict[ID][self.headers[i]][index][:7])- datetime.datetime(1900,1,1)).total_seconds()))
                        #current_values.append(time.mktime(self.patient_dict[ID][self.headers[i]][index]))
                    else:
                        current_values.append(self.patient_dict[ID][self.headers[i]][index])
                daily_set = np.append(daily_set, np.column_stack(current_values), axis=0)
                index += 1
                prev_time_point = copy.deepcopy(curr_time_point)

                # If this is the last one, we should stop.
                if index == len(self.patient_dict[ID]['time']):
                    [attr, aggregated_values] = self.aggregate_day(extended_headers, daily_set, day_count, type)
                    if len(new_attributes) == 0:
                        new_attributes = attr
                        aggr_set = np.zeros((0, len(attr)))
                    aggr_set = np.append(aggr_set, np.column_stack(aggregated_values), axis=0)

            for r in range(row_count, aggr_set.shape[0]):
                self.writer.writerow(aggr_set[r,:])
            row_count = aggr_set.shape[0]

        # print aggr_set.shape
        return new_attributes, aggr_set


    # Aggregate the data in a way which depends on the type.
    def aggregate_data(self, type):
        # This allows the aggregation of data into learnable sets, depending on the type this can be
        # per day, etc.

        if type == 0:
            return self.aggregate(type)
        elif type == 1:
            return self.aggregate(type)
        #else:
            # Not implemented yet.....
        return

    # Filter values based on the predifined min and max values.
    def min_max_filter(self, feature, value):

        if self.ranges.has_key(feature) and not value == '':
            if float(value) < self.ranges[feature]['min_value'] or float(value) > self.ranges[feature]['max_value']:
                return ''
        return value


    # Type defines the type of datasets created:
    # - 0 for SDAS (cf. thesis Caleb, daily summed over days)
    # - 1 for DASn (again following thesis Caleb, daily, days are independent)

    def read_files_and_calculate_attributes(self, file, file_out, type=0):

        self.writer = io.write_csv(file_out)
        complete_patient_ids = self.determine_complete_patient_ids(file)
        len(complete_patient_ids)
        print('====== reading the data')
        rows = io.read_csv(file, ',')
        print('====== pointer to data obtained')

        counter = 0
        ids = []
        dataset_headers = []

        for row in rows:
                #for row in rows:
                if counter % 10000 == 0:
                    print('====== ' + str(counter))
                # Assuming the headers are in the first two rows and the label in the last.
                if counter == 0:
                    temp_dataset_headers = row[2:len(row)]

                    # Create all headers, also of derived categorical attributes
                    # attributes over time and derivations of multiple attributes combined
                    # will be derived later.

                    for header in temp_dataset_headers:
                        header = header.lower()
                        dataset_headers.append(header)
                        self.headers.append(header)

                elif  str(row[0] + row[1]) in complete_patient_ids:
                    # Assuming ID is the first combined with the second attribute (patient ID plys stay ID.
                    id = str(row[0] + row[1])
                    if id not in ids :
                        ids.append(id)
                        self.patient_dict[id] = {}
                        for header in self.headers:
                            self.patient_dict[id][header] = []

                    # Get the time to order based upon it
                    # Time had this structure in original csv: 07-SEP-82 07.30.00.000000000 PM US/EASTERN
                    # Time had this structure now :2682-09-07 18:3
                    #this was changed by Ali because of the new dataset
                    timestamp = time.strptime(row[self.headers.index('time')+2][0:16],"%Y-%m-%d %H:%M")
                    # timestamp = time.strptime(time.strftime('%d-%b-%y %H.%M', timestamp),'%d-%b-%y %H.%M', )

                    times = self.patient_dict[id]['time']
                    # Currently no ordering of the times assumed. If they are, just append at the end
                    index = 0
                    while index < len(times) and times[index] < timestamp:
                        index += 1
                    for row_index in range(2, len(row)):
                        if dataset_headers[row_index-2] == 'time':
                            self.patient_dict[id]['time'].insert(index, timestamp)
                        else:
                            # Determine the values (there can be multiple in the case of categorical attributes)
                            [features, values] = [[dataset_headers[row_index-2]], [row[row_index]]]
                            for i in range(0, len(values)):
                                self.patient_dict[id][features[i]].insert(index, values[i])
                counter += 1
        print(self.patient_dict.keys())
        print(len(self.patient_dict.keys()))
        return self.aggregate_data(type)
