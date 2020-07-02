import argparse
from sys import exit, argv, stderr
from os import listdir, replace
from os.path import isfile, isdir
import yaml


def merge_csv_files(target_dir):
    analysis_dir_prefix = 'ANALYSIS'
    single_file_name = 'statistics.csv'
    merged_file_name = 'statistics_merged.csv'

    analysis_dirs = [d for d in sorted(listdir(target_dir)) if d.startswith(analysis_dir_prefix)]

    if len(analysis_dirs) == 0:
        print("OWIER")
        return

    if isfile("{}/{}".format(target_dir,
                             merged_file_name)):
        replace("{}/{}".format(target_dir, merged_file_name),
                "{}/{}.old".format(target_dir, merged_file_name))

    with open("{}/{}".format(target_dir,
                             merged_file_name), 'w') as merged_csv_file:
        stats_file = open("{}/{}/{}".format(target_dir,
                                            analysis_dirs[0],
                                            single_file_name), 'r')
        header, content = stats_file.readlines()
        merged_csv_file.write(header)
        merged_csv_file.write(content)

        stats_file.close()
        
        for i in range(1, len(analysis_dirs)):
            try:
                stats_file = open("{}/{}/{}".format(target_dir,
                                                    analysis_dirs[i],
                                                    single_file_name), 'r')
                this_header, content = stats_file.readlines()
                if not this_header == header:
                    print("The headers of the csv files in {} and {} do not match.  Please re-create the csv files.".format(analysis_dirs[0], analysis_dirs[i]))
                    return
                merged_csv_file.write(content)
            except:
                print(analysis_dirs[i])


        stats_file.close()

    merged_csv_file.close()

def main():
    target_dir = argv[1]

    if not isdir(target_dir):
        print("Error. Analysis directory {} not found.".format(target_dir))
        exit()
    
    merge_csv_files(target_dir)
    
if __name__ == "__main__":
    if len(argv) == 1:
        print("Please specify the directory in which to look for analysis dirs.")
        exit()
    exit(main())
