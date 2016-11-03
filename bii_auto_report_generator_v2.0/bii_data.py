import numpy as np
import pandas as pd



def bii_data(group_type,code):


	# Read in data
    df0 = pd.read_json('biidata.json')

    # Replace empty cells with numpy NaN
    df0=df0.replace('', np.nan, regex=True)

    # Convert columns to floating points if possible
    for col in df0:
        try:
            df0[col] = df0[col].astype('float') #convert data to float if possible
        except:
            pass

    # Set all codes to lower case
    df0['PARTNER CODE'] = df0['PARTNER CODE'].str.lower()
    df0['PROJECT CODE'] = df0['PROJECT CODE'].str.lower()


    # Set user input to lower case, and check if we have Partner + Project Code
    if group_type == 'projAndPart':
        code[0] = code[0].lower()
        code[1] = code[1].lower()
        
    else:
        code = code.lower()
    

    # identify what type of data to collect

    if group_type == 'part':
        df0 = df0.loc[df0['PARTNER CODE'] == code]
    elif group_type == 'proj':
        df0 = df0.loc[df0['PROJECT CODE'] == code]
    elif group_type == 'projAndPart':
        df0 = df0.loc[df0['PARTNER CODE'] == code[0]]
        df0 = df0.loc[df0['PROJECT CODE'] == code[1]]
    # Comparison
    elif group_type == 'comp':
        if code == 'male':
            df0 = df0.loc[df0['GENDER'] == 2]
        elif code == 'female':
            df0 = df0.loc[df0['GENDER'] == 1]
        elif code == 'study1':
            df0 = df0.loc[df0['STUDY'] == 1]
        elif code == 'study2':
            df0 = df0.loc[df0['STUDY'] == 2]
        elif code == 'study3':
            df0 = df0.loc[df0['STUDY'] == 3]
        elif code == 'all':
            group_type = 'comp'
        else:
            df0 = df0.loc[df0['PARTNER CODE'] == code]
            df1 = df0.loc[df0['PROJECT CODE'] == code]
            df0.append(df1)


    return df0



