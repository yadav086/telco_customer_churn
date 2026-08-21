def prepare_feature(df):
   df_num = df.select_dtypes(include ='number')
   df_cat = df.select_dtypes(exclude ='number').drop('Churn_Label',axis =1)
   all_features = [*df_cat.columns, *df_num.columns]
   X= df[all_features]
   y = df['Churn_Label'].map({'Yes':0,'No':1})
   
   return X,y, df_num, df_cat 