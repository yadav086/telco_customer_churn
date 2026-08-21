from  load_config import load_config

def clean_df(df):
  config = load_config('config.yaml')
  remove_col = config['data']['remove_columns']	
  
  df['Total Charges']= df['Total Charges'].replace(' ', 0)
  df['Total Charges'] =df['Total Charges'].astype('float')
   
  #df.columns = [col.strip().replace(' ', '_') for col in df.columns]
  df1= df.copy()
  remove_columns= remove_col
  df=df.drop(remove_columns,axis =1)		
  
  return df