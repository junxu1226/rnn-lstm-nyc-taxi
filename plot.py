import pandas as pd
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
#Inline Plotting for Ipython Notebook
#%matplotlib inline

df=pd.io.gbq.read_gbq("""
SELECT ROUND(pickup_latitude, 4) as lat, ROUND(pickup_longitude, 4) as long, COUNT(*) as num_pickups
FROM [nyc-tlc:green.trips_2015]
WHERE (pickup_latitude BETWEEN 40.81 AND 40.91) AND (pickup_longitude BETWEEN -73.96 AND -73.77 )
GROUP BY lat, long
""", project_id='taxiproject-146322')


pd.options.display.mpl_style = 'default' #Better Styling
#matplotlib.style.use('ggplot')
new_style = {'grid': True, 'facecolor': 'blue'} #Remove grid
mpl.rc('axes', **new_style)

from matplotlib import rcParams
rcParams['figure.figsize'] = (25, 20) #Size of figure
rcParams['figure.dpi'] = 250



P=df.plot(kind='scatter', x='long', y='lat',color='white',xlim=(-73.96,-73.77),ylim=(40.81, 40.91),s=.02,alpha=.6)
P.set_axis_bgcolor('black') #Background Color
fig = P.get_figure()
fig.savefig("output.png")
