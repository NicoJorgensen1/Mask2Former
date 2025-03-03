import os 

# Define function to log information about the dataset
def printAndLog(input_to_write, logs, print_str=True, write_input_to_log=True, prefix="\n", postfix="", oneline=True, length=None):
    mode = "a" if os.path.isfile(logs) else "w"                                                     # Whether or not we are appending to the logfile or creating a new logfile
    logs = open(file=logs, mode=mode)                                                               # Open the logfile, i.e. making it writable
    
    if isinstance(input_to_write, str):                                                             # If the input is of type string ...
        if print_str: print(input_to_write)                                                         # ... and it needs to be printed, print it ...
        if write_input_to_log: logs.writelines("{:s}{:s}{:s}".format(prefix, input_to_write, postfix))  # ... if it needs to be logged, write it to log
    if isinstance(input_to_write, list):                                                            # If the input is a list ...
        if print_str:                                                                               # ... and the list needs to be printed ...
            for string in input_to_write: print(string, end="\t" if oneline else "\n")              # ... print every item in the list
        if write_input_to_log:                                                                      # ... if the input needs to be logged ...
            if write_input_to_log: logs.writelines("{:s}".format(prefix))                           # ... the prefix is written ...
            for string in input_to_write: logs.writelines("{}\n".format(string))                    # ... and then each item in the list is written on a new line ...
            logs.writelines("{:s}".format(postfix))                                                 # ... and then the postfix is written
    if isinstance(input_to_write, dict):                                                            # If the input is a dictionary ...
        if write_input_to_log: logs.writelines("{:s}".format(prefix))                               # ... and the input must be logged, the prefix is written ...
        if write_input_to_log or print_str:                                                         # ... or if the input has to be either logged or printed ...
            for key, val in input_to_write.items():                                                 # ... and then we loop over all key-value pairs in the dictionary ...
                if isinstance(val, float): val = "{:.4f}".format(val)                               # If the value is a float value, round it to four decimals
                else: val = str(val)                                                                # Else, simply convert the value to a string
                if val == "": val = "None"                                                          # If the value is empty, let it be "None"
                key = key + ": "                                                                    # Add the colon to the key
                if length != None: key = key.ljust(length)                                          # If any length has been chosen by the user, pad the key-name string to that specific length
                string = "{:s}{:s}{:s}".format(key, val.ljust(9), "||".ljust(5) if oneline else "\n")   # Create the string to print and log
                if write_input_to_log: logs.writelines(string)                                      # ... If the string needs to be logged, log the key-value pairs on the same line ...
                if print_str: print(string, end="\t" if oneline else "")                            # ... and print it if chosen ...
        if write_input_to_log: logs.writelines("{:s}".format(postfix))                              # ... before writing the postfix 
    
    # Close the logfile again
    logs.close()                                                                                    # Close the logfile again
