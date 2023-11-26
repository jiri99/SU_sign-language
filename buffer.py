from processing import ParquetProcess

class ParquetData:
  def __init__(self):
    self.data = {}

  def read_all(self, path, df_list, landmark_id, max_length = 140):
    for participant_folder in os.listdir(path + "/train_landmark_files/"):
      path_folder = path + "/train_landmark_files/" + participant_folder + "/"
      for file_name in os.listdir(path_folder):
        file_num = file_name.split(".")[0]
        df_row = df_list[(df_list.participant_id == int(participant_folder)) & (df_list.sequence_id == int(file_num))]
        file_path = path + "/train_landmark_files/" + participant_folder + "/" + file_num + ".parquet"
        if(df_row.length_frames.values[0] > max_length):
          print('File was skipped:: ', file_path)
          continue
        else:
          sign_name = df_row.sign.values[0]
          if(not sign_name in list(self.data.keys())):
            self.data[sign_name] = {}
          self.data[sign_name][participant_folder] = ParquetProcess(file_path, landmark_id, max_length)
    print("Data read completed!")
