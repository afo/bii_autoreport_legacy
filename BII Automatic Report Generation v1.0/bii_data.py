#import MySQLdb
import numpy as np
import pandas as pd
#import pandas.io.sql as psql
#from MySQLdb.converters import conversions
#from MySQLdb.constants import FIELD_TYPE



def bii_data(ident,code):


    #conversions[FIELD_TYPE.DECIMAL]=float
    #conversions[FIELD_TYPE.NEWDECIMAL]=float

    #conn = MySQLdb.connect(host="bii.catx38sk4s1o.us-west-2.rds.amazonaws.com", user="awsuser", passwd="awspassword", db="bii_db")
    
    #sql = "select * from bii_data"

    #df0 = psql.read_frame(sql,conn)

    df0 = pd.read_json('biidata.json')
    df0=df0.replace('', np.nan, regex=True)




    #df0=df0.replace('', np.nan) # replace empty strings BUG

    df0=df0.replace('', np.nan, regex=True)

    #for c in df0:
    #    df0[c] = df0[c].replace('', np.nan)

    for col in df0:
        try:
            df0[col] = df0[col].astype('float') #convert data to float if possible
        except:
            pass

    df0['PARTNER CODE'] = df0['PARTNER CODE'].str.lower()

    df0['PROJECT CODE'] = df0['PROJECT CODE'].str.lower()


    if ident == 6:
        code[0] = code[0].lower()
        code[1] = code[1].lower()
        
    else:
        code = code.lower()
    
    if ident == 1:
        df0 = df0.loc[df0['Email Address'] == code]
    elif ident== 2:
        df0 = df0.loc[df0['PARTNER CODE'] == code]
    elif ident == 5:
        df0 = df0.loc[df0['PROJECT CODE'] == code]
    elif ident == 6:
        df0 = df0.loc[df0['PARTNER CODE'] == code[0]]
        df0 = df0.loc[df0['PROJECT CODE'] == code[1]]
        #df1 = df0.loc[df0['PROJECT CODE'] == code[1]]
        #df0.append(df1)
    elif ident == 3:
        if code == 'male':
            df0 = df0.loc[df0['Gender'] == 2]
        elif code == 'female':
            df0 = df0.loc[df0['Gender'] == 1]
        elif code == 'study1':
            df0 = df0.loc[df0['Study'] == 1]
        elif code == 'study2':
            df0 = df0.loc[df0['Study'] == 2]
        elif code == 'study3':
            df0 = df0.loc[df0['Study'] == 3]
        elif code == 'all':
            ident = 3
        else:
            df0 = df0.loc[df0['PARTNER CODE'] == code]
            df1 = df0.loc[df0['PROJECT CODE'] == code]
            df0.append(df1)





    #if ident == 1:
    #    df0 = df0.loc[df0['email'] == code]
    #elif ident ==2:
    #    df0 = df0.loc[df0['wg_code'] == code]
        
    #Trust
    qt1_std = 1.013
    qt1_mean = 3.2
    qt1_weight = 0.67

    qt4_std = 1.1
    qt4_mean = 3.1
    qt4_weight = 0.33

    ET1 = (df0['QT1'].fillna(value=3) - qt1_mean)/qt1_std
    ET4 = -(df0['QT4'].fillna(value=3) - qt4_mean)/qt4_std

    trust = 5.5 + 2*ET1*qt1_weight + 2*ET4*qt4_weight
    if np.mean(trust) > 10:
        trust = 10
    elif np.mean(trust) < 1:
        trust = 1
    df0['trust'] = trust

    #Resilience

    qf2_std = 0.82
    qf2_mean = 4.2
    qf2_weight = 0.18

    qf3_std = 0.69
    qf3_mean = 4.4
    qf3_weight = .82

    EF2 = (df0['QF2'].fillna(value=3) - qf2_mean)/qf2_std
    EF3 = (df0['QF3'].fillna(value=3) - qf3_mean)/qf3_std

    res = 5.5 + 2*EF2*qf2_weight + 2*EF3*qf3_weight
    if np.mean(res) > 10:
        res = 10
    elif np.mean(res) < 1:
        res = 1
    df0['res'] = res

    qd2_std = 0.93
    qd2_mean = 3.96
    qd2_weight = 0.45

    qd3_std = 0.91
    qd3_mean = 4.1
    qd3_weight = .55

    ED2 = (df0['QD2'].fillna(value=3) - qd2_mean)/qd2_std
    ED3 = (df0['QD3'].fillna(value=3) - qd3_mean)/qd3_std

    div = 5.5 + 2*ED2*qd2_weight + 2*ED3*qd3_weight
    if np.mean(div) > 10:
        div = 10
    elif np.mean(div) < 1:
        div = 1
    df0['div'] = div

    qb2_std = 0.73
    qb2_mean = 4.2
    qb2_weight = 0.89

    qb3_std = 0.84
    qb3_mean = 3.4
    qb3_weight = .11

    EB2 = (df0['QB2'].fillna(value=3) - qb2_mean)/qb2_std
    EB3 = (df0['QB3'].fillna(value=3) - qb3_mean)/qb3_std

    bel = 5.5 + 2*EB2*qb2_weight + 2*EB3*qb3_weight
    if np.mean(bel) > 10:
        bel = 10
    elif np.mean(bel) < 1:
        bel = 1
    df0['bel'] = bel

    qc4_std = 0.95
    qc4_mean = 3.38
    qc4_weight = 1

    EC4 = (df0['QC4'].fillna(value=3) - qc4_mean)/qc4_std

    col = 5.5 + 2*EC4*qc4_weight

    if np.mean(col) > 10:
        col = 10
    elif np.mean(col) < 1:
        col = 1

    df0['col'] = col

    qp3_std = 1.11
    qp3_mean = 3.12
    qp3_weight = 1


    EP3 = -(df0['QP3'].fillna(value=3) - qp3_mean)/qp3_std

    resall = 5.5 + 2*EP3*qp3_weight
    if np.mean(resall) > 10:
        resall = 10
    elif np.mean(resall) < 1:
        resall = 1
    df0['resall'] = resall


# THESE LAST ONES NEED AN UPDATE AS WELL!!! ARE WE SURE WE GOT COMFORT/CZX RIGHT?

    czx=df0['CZX']= df0['CZ'].fillna(value=2.5)
    cz=(df0['CZ'].fillna(value=2.5)-1)*3+1
    sdr = (df0['SDR'].fillna(value=3))

    CZn = 2*(czx-2)*0.8
    SDRn = 2*(sdr-4)*0.2
    iz=df0['IZ']= 5.5 + CZn + SDRn

    if np.mean(iz) > 10:
        iz = 10
    elif np.mean(iz) < 1:
        iz = 1
    df0['IZ'] = iz

    # if ident == 1:
    #     if czx == 4:
    #         msg = "Based on your comfort with ambiguity, your MINDSET LEANS STRONGLY towards INNOVATION. If you have interest in operational innovation, you should pre-analyze situations and focus more on risk mitigation."
    #     elif czx == 3:
    #         msg = "Based on your comfort with ambiguity, your MINDSET covers both operations and innovation, but LEANS towards INNOVATION. If you have interest in operational innovation and precision, you should pre-analyze situations and focus more on risk mitigation."
    #     elif czx == 2:
    #         msg = "Based on your comfort with ambiguity, your MINDSET covers both operations and innovation but LEANS towards OPERATIONAL INNOVATION. If you are interested in innovation or entrepreneurship, you should try to grow by increasing your comfort even when you are in areas where you are not knowledgeable. Also look into techniques that reduce fears."
    #     elif czx == 1:
    #         msg = "Based on your comfort with ambiguity, your MINDSET LEANS STRONGLY towards OPERATIONAL INNOVATION and PRECISION. If you are interested in innovation or entrepreneurship, you should increase your comfort when you are in uncertain situations. Also look into techniques that reduce fears."



    
    #Calculate overall score
    score=df0['score']=(df0['trust']+df0['res']+df0['div']+df0['bel']+df0['col']+df0['resall']+df0['IZ'])/7
    
    return [trust,res,div,bel,col,resall,czx,cz,iz,score]

