
# coding: utf-8

# # The Floow Technical challenge
# <br>
# 

# In[133]:


#----------------------------------------------------------------------------------------------
path =r'C:\Users\Kyled\Downloads\all journeys - Document Classification_ Sensitive\all journeys' 
outfile=r'C:\Users\Kyled\Downloads\all journeys - Document Classification_ Sensitive\output.csv'
#----------------------------------------------------------------------------------------------


# ## Libraries

# In[134]:



#-------------------------------------------------
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
import seaborn as sns
from scipy import stats
import scipy.stats as st
import warnings
import os
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.20f}'.format
pd.options.mode.chained_assignment = None
#--------------------------------------------------


# ## Functions

# In[135]:


#------------------------------
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def f(col):
    if col['M_A'] ==1 & col['B']==1:
        val = 'Accident'
    
    else:
        val = 'N'
    return val
#--------------------------------


# ## Wrangle

# In[136]:



#---------------------------------------------------------------------
files = glob.glob(path +'/*.csv') 

df = pd.concat([pd.read_csv(fp).assign(File_ref=os.path.basename(fp)) for fp in files])
r, c = df.shape
print("Sum of rows from concatenation:\n"
      "Total rows =" ,r,"\nTotal columns =",c)
#----------------------------------------------------------------------


# In[137]:


df.head
df.describe()


# ## Cleansing
# 
# ### Time

# In[138]:


df['DateTime'] = pd.to_datetime(df.iloc[:,0].combine_first(df.timestamp), unit='ms')
df['Date'] = df['DateTime'].dt.date
df['Unit_check']=(np.abs(df['x'])+np.abs(df['y'])+np.abs(df['z']))/3
table=pd.pivot_table(df,index=["File_ref"])
dfb = table[table['Unit_check']>2].index.values.astype(str)[0]
print("Files with units in m s^-1 :",dfb)


# ### Acceleration 
# 
# Magnitude of acceleration vector since we have no fixed orientation 
# 
# \begin{align}
# \ | a\vec\ |=\sqrt{a_x^2+a_y^2+a_z^2}
# \end{align}

# In[139]:


s=df[df['File_ref'].str.contains(dfb)]
df=df[~df['File_ref'].str.contains(dfb)]
df['Mag/G']=np.sqrt(df.x**2+df.y**2+df.z**2)
df['Mag/G'].fillna(0, inplace=True)

s['Mag/G']=np.sqrt(s.x**2+s.y**2+s.z**2)/9.81
s['Mag/G'].fillna(0, inplace=True)

df=df.append(s)


# ### Units 

# In[140]:


df.rename({'bearing': 'bearing/deg', 'height': 'height/m',
           'speed': 'speed m s^-1 (scalar)'}, inplace=True,axis=1)
df['Speed km h^-1']=df['speed m s^-1 (scalar)']*3.6 # m s^-1 is the SI unit, but km h^-1 helpful


# In[141]:


#Copy of df with relevant columns 
output= pd.DataFrame(df[['Date','DateTime', 'Mag/G', 'Speed km h^-1','height/m',
                         'bearing/deg','lat', 'lon','File_ref']].copy()
)


# To construct a suitable metric we will need to fill values for the speed.
# I chose to fill with the average between values dynamically because both forward propogation and column wide mean are not appropriate.

# In[142]:


output['Speed km h^-1']=output['Speed km h^-1'].interpolate()


# ### Understanding the data

# ##### Speed

# In[143]:


sns.set_style('darkgrid')
sns.distplot(output['Speed km h^-1'].dropna(),bins=10)
output['Speed km h^-1'].describe()


# Data is heavily skewed towards lower speeds,which is consistent with reality.Data will need to be normalised to score appropriatley.Mean here is ~30 km h^-1

# #### Acceleration

# In[144]:


#-------------
x = df.DateTime
x_accel = df.x
y_accel = df.y
z_accel = df.z
#--------------------
plt.subplot(3, 1, 1)
plt.plot(x, x_accel, '.-')
plt.title('Vector Acceleration/G')
plt.ylabel('X acceleration')
plt.xticks([])
plt.subplot(3, 1, 2)
plt.plot(x, y_accel, '.-')
plt.xlabel('time')
plt.ylabel('Y acceleration')
plt.xticks([])
plt.subplot(3, 1, 3)
plt.plot(x, z_accel, '.-')
plt.xlabel('time')
plt.ylabel('Z acceleration')
plt.xticks([])

plt.show()


# In[145]:


plt.xlim(1, 10)
sns.distplot(output['Mag/G'].dropna(),bins=50)
np.mean(output['Mag/G'])


# Again we have a heavy skew towards lower accelerations. This is consistent with the bulk of driving (i.e no events). Normalisation to be applied.

# ### Process

# $AD: x -> \{0,1\}$
# 
# (1) a high acceleration event whilst the vehicle is moving
# above the threshold speed, $M_S$.
# 
# 
# $$AD (x,y)=
# \begin{cases}
# (M_A,β)==1,\\
# \; 0
# \end{cases}
# $$
# 
# 
# 

# First we consider acceleration events from the phone alone i.e false positives 
# Here we set $M_A$, the minimum acceleration for an acceleration event to trigger accident detection.
# 
# $M_A = 4$
# 
# 
# 

# In[146]:


output['M_A'] = np.where(output['Mag/G']<4, 0, 1)
output['M_A'].value_counts()


# $M_S$ is the minimum speed in order to activate the accident detection system and serves as a low pass filter to rule out other activities and ensure the phone is in the vehicle.
# 
# $β$ is a speed threshold variable with value 1 if the phone has been traveling at greater than $M_S$.
# 
# $M_S = 25$ 

# In[147]:


M_S = 25
output['B'] = np.where(output['Speed km h^-1']>M_S, 1, 0)
output['B'].value_counts()


# Here we apply the Accident detection function f. That is, 1 if $$Mag>M_A, B==1$$  
# 
# See functions.

# In[148]:


output['AD'] = output.apply(f, axis=1)
output['AD'].value_counts()


# Next we establish a time of the event occuring. A time stamp will be generated for all accidents that meet the criteria. A decision during filtering will need to be made with respect to when we register the event.

# In[149]:


output['Time_of_event'] =output.apply(lambda x: '-' if x['AD'] == 'N' else x['DateTime'],axis=1) 


# ## Scoring 

# The variables of interest were highly skewed.
# With respect to outlier treatment, we wish to keep these since they correspond to high speed,high acceleration events.
# Z scoring,logarithmic treatment and sigmoid functions introduced to much distortion and unreliable scoring at extremes.
# I opted for percentile linearisation.

# In[150]:


size = len(output['Mag/G'])-1
output['Mag_pl']= output['Mag/G'].rank(method='min').apply(lambda x: (x-1)/size)
output['Mag_pl'].describe()

size = len(output['Speed km h^-1'])-1
output['Speed_pl']= output['Speed km h^-1'].rank(method='min').apply(lambda x: (x-1)/size)
output['Speed_pl'].describe()


# Next, we take a weighted average of the two scores.
# 
# we set weights for speed and acceleration;
# $w_A=1$
# $w_B=2$ respectivley.
# 
# $W = \frac{w_A}{w_A+w_B} \cdot  Z_A + \frac{w_B}{w_A+w_B} \cdot Z_B$
# 
# Divide the Result by its standard deviation to give a severity index SI.
# 
# And introduce a column for confidence.

# In[151]:


w_A=1
w_B=2
output['W']=(w_A/(w_A+w_B))*output['Mag_pl']+(w_B/(w_A+w_B))*output['Speed_pl']
output['SI']=output['W']/np.std(output['W'])
output['conf']=(st.norm.cdf(output['SI']))


# Next we produce a table with accident flags,and then output back to the original journeys.
# 

# In[152]:


a=output.loc[output['AD'] == 'Accident']


# In[167]:


for i, x in a.groupby('File_ref'):
    p = os.path.join(os.getcwd(), "data_{}.csv".format(i.lower()))
    x.to_csv(p, index=False)
csvs=[x for x in os.listdir() if x.endswith('.csv')]
fns=[os.path.splitext(os.path.basename(x))[0] for x in csvs]
d={}


# In[188]:


print("Journeys with accident flags: \n",a.File_ref.unique(), "\n")

v=a.AD.count()
table=pd.pivot_table(a,index=["File_ref",""])
print("Total accident flags: ",a.AD.count(),"\n")

print("The most severe accidents for each Journey were:")
#display(table)

    
    

