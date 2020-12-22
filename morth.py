#!/usr/bin/env python
# coding: utf-8

# bokeh serve --show morth.py

# In[50]:


import math
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from math import *

from bokeh import *
from bokeh.plotting import * 
from bokeh.models import * 
from bokeh.models.tools import *
from bokeh.transform import *
from bokeh.palettes import *
from bokeh.embed import *
from bokeh.layouts import *
from bokeh.io import *
from bokeh.models.widgets import *

import warnings 
warnings.filterwarnings('ignore')

output_notebook()


# In[51]:


df=pd.read_csv('roads.csv')
df['accidents_per_km']=df.Accidents/df.Kms
df['fatality_rate']=df['killed']/(df['injured']+df['killed'])
df['odds_survival']=(1-df['fatality_rate'])/df['fatality_rate']
source = ColumnDataSource(df)

roads_list = source.data['roads'].tolist()

# Add plot
p1 = figure(
    plot_width=500,
    plot_height=500,
    title='Accidents across category of roads for year 2019',
    x_axis_label='Number of accidents per kilometre',
    y_axis_label='Odds of survival',
    tools="box_select,save,reset"
)

# Render glyph
p1.circle(y='odds_survival', x='accidents_per_km', 
         fill_color=factor_cmap('roads', palette=Spectral6, factors=roads_list),
         source=source, legend='roads',radius=0.025)

# Add Legend
p1.legend.orientation = 'vertical'
p1.legend.location = 'top_right'
p1.legend.label_text_font_size = '10px'

# Add Tooltips
hover = HoverTool()
hover.tooltips = """
  <div>
    <h3>@roads</h3>
    <div><strong>Accidents: </strong>@Accidents</div>
    <div><strong>Killed: </strong>@killed</div>
    <div><strong>Injured: </strong>@injured</div>
  </div>
"""
p1.add_tools(hover)


# In[52]:


t1 = Div(text="""<b>Interpretation:</b> As the accident rate increases, the odds of survival decrease. There is a significant difference in accident rates across roads. The accident rate for national highways are the highest. """, width=500, height=70)
c1=column(p1,t1)
show(c1)


# In[53]:


df=pd.read_csv('heatmap_india.csv')
source = ColumnDataSource(df)
states_list = source.data['states'].tolist()

# Add plot
p21 = figure(
    plot_width=600,
    plot_height=600,
    x_axis_label='Accidents per lakh population',
    y_axis_label='Survival probability',
    tools="box_select,save,reset"
)

color_palette=Turbo256[::-1]
color_mapper = LinearColorMapper(palette=color_palette, low=df['survival_rate'].min(), high=df['survival_rate'].max())

# Render glyph
p21.circle(y='survival_rate', x='accidents_per_lakh_population', fill_color=transform('survival_rate',color_mapper),          source=source, radius=1.5)

# Add Tooltips
hover = HoverTool()
hover.tooltips = """
  <div>
    <h3>@states</h3>
    <div><strong>Killed: </strong>@killed</div>
    <div><strong>Injured: </strong>@injured</div>
  </div>
"""

p21.add_tools(hover)


# In[54]:


df=pd.read_csv('heatmap_india.csv')
source = ColumnDataSource(df)
states_list = source.data['states'].tolist()

# Add plot
p22 = figure(
    plot_width=600,
    plot_height=600,
    x_axis_label='Accidents per 10,000 vehicles',
    y_axis_label='Survival probability',
    tools="box_select,save,reset"
)

color_palette=Turbo256[::-1]
color_mapper = LinearColorMapper(palette=color_palette, low=df['survival_rate'].min(), high=df['survival_rate'].max())

# Render glyph
p22.circle(y='survival_rate', x='accidents_per_10,000_vehicles', fill_color=transform('survival_rate',color_mapper),          source=source, radius=0.35)

# Add Tooltips
hover = HoverTool()
hover.tooltips = """
  <div>
    <h3>@states</h3>
    <div><strong>Killed: </strong>@killed</div>
    <div><strong>Injured: </strong>@injured</div>
  </div>
"""

p22.add_tools(hover)


# In[55]:


df=pd.read_csv('heatmap_india.csv')
source = ColumnDataSource(df)
states_list = source.data['states'].tolist()

# Add plot
p23 = figure(
    plot_width=600,
    plot_height=600,
    x_axis_label='Accidents per lakh population',
    y_axis_label='Severity rate',
    tools="box_select,save,reset"
)

color_palette=Turbo256
color_mapper = LinearColorMapper(palette=color_palette, low=df['severity_rate'].min(), high=df['severity_rate'].max())

# Render glyph
p23.circle(y='severity_rate', x='accidents_per_lakh_population', fill_color=transform('severity_rate',color_mapper),          source=source, radius=1.5)

# Add Tooltips
hover = HoverTool()
hover.tooltips = """
  <div>
    <h3>@states</h3>
    <div><strong>Killed: </strong>@killed</div>
    <div><strong>Injured: </strong>@injured</div>
  </div>
"""

p23.add_tools(hover)


# In[56]:


df=pd.read_csv('heatmap_india.csv')
source = ColumnDataSource(df)
states_list = source.data['states'].tolist()

# Add plot
p24 = figure(
    plot_width=600,
    plot_height=600,
    x_axis_label='Accidents per 10,000 vehicles',
    y_axis_label='Severity rate',
    tools="box_select,save,reset"
)

color_palette=Turbo256
color_mapper = LinearColorMapper(palette=color_palette, low=df['severity_rate'].min(), high=df['severity_rate'].max())

# Render glyph
p24.circle(y='severity_rate', x='accidents_per_10,000_vehicles', fill_color=transform('severity_rate',color_mapper),          source=source, radius=0.35)

# Add Tooltips
hover = HoverTool()
hover.tooltips = """
  <div>
    <h3>@states</h3>
    <div><strong>Killed: </strong>@killed</div>
    <div><strong>Injured: </strong>@injured</div>
  </div>
"""

p24.add_tools(hover)


# In[57]:


t21 = Div(text="""<b>Interpretation:</b> Goa has the highest accident rate (accidents per lakh population) and Punjab has the lowest survival probability. Some of the north-eastern states and union territories have low accident rates and high survival probabilities.""", width=600, height=50)
t22 = Div(text="""<b>Interpretation:</b> Madhya Pradesh has the highest accident rate (accidents per 10,000 vehicles).""", width=600, height=50)
t23 = Div(text="""<b>Interpretation:</b> Tamil Nadu has the lowest severity rate.""", width=600, height=50)
t24 = Div(text="""<b>Interpretation:</b> """, width=600, height=50)

c21 = column(p21,t21)
c22 = column(p22,t22)
c23 = column(p23,t23)
c24 = column(p24,t24)
g1=gridplot([[c21,c23],[c22,c24]])
show(g1)


# In[58]:


df=pd.read_csv('roads_rates.csv')
source = ColumnDataSource(df)
years_list = source.data['Year'].tolist()
roads_list=['National Highways','State Highways','Other Roads','Total']
categories=['Accidents','Persons Killed','Persons Injured']

chosen_idx=[0,1,3,5]
colors=[Spectral6[i] for i in chosen_idx]
cols=list(df.columns)

# Add Tooltips
hover1 = HoverTool(names=['National Highways'])
hover1.tooltips = [('Rate', '@{NH_road_accidents}')]

hover2 = HoverTool(names=['State Highways'])
hover2.tooltips = [('Rate', '@{SH_road_accidents}')]

hover3 = HoverTool(names=['Other Roads'])
hover3.tooltips = [('Rate', '@{OR_road_accidents}')]

hover4 = HoverTool(names=['Total'])
hover4.tooltips = [('Rate', '@{Total_road_accidents}')]

p3 = figure(
    plot_width=500,
    plot_height=450,
    title='Accidents per km of road across years',
    x_axis_label='Year',
    y_axis_label='Rate (Count/km)',
    tools=["box_select,save,reset",hover1,hover2,hover3,hover4]
)

p3.xaxis.ticker = years_list

f=0
for i, j in zip([f+1,f+4,f+7,f+10], range(4)):
    p3.line(y=cols[i], x='Year', source=source, line_color=colors[j], line_width=4, line_alpha=0.6, legend=roads_list[j])
    p3.asterisk(y=cols[i], x='Year', source=source, color='black', name=roads_list[j], size=7)
    
p3.legend.orientation = 'vertical'
p3.legend.location = 'top_right'
p3.legend.label_text_font_size = '10px'

drop_bar = Select(options=categories, value=categories[0])


# In[59]:


t3 = Div(text="""<b>Interpretation:</b> Number of accidents per kilometre, persons killed per kilometre and persons injured per kilometre have decreased over time for National Highways, while they remain roughly same for State Highways, Other Roads and Overall.""", width=500, height=50)
c3=column(drop_bar,p3,t3)
show(c3)


# In[60]:


def update_bar_chart(attrname, old, new):
    
    if drop_bar.value==categories[0]:
        f=0
        label='road_accidents'
        
    elif drop_bar.value==categories[1]:
        f=1
        label='persons_killed'
        
    else:
        f=2
        label='persons_injured'
        
    df=pd.read_csv('roads_rates.csv')
    source = ColumnDataSource(df)
    cols=list(df.columns)
    
    label1=f'NH_{label}'
    label2=f'SH_{label}'
    label3=f'OR_{label}'
    label4=f'Total_{label}'
        
    chosen_idx=[0,1,3,5]
    colors=[Spectral6[i] for i in chosen_idx] 
    roads_list=['National Highways','State Highways','Other Roads','Total']
    
    # Add Tooltips
    hover1 = HoverTool(names=['National Highways'])
    hover1.tooltips = [('Rate', f"@{label1}")]

    hover2 = HoverTool(names=['State Highways'])
    hover2.tooltips = [('Rate', f"@{label2}")]

    hover3 = HoverTool(names=['Other Roads'])
    hover3.tooltips = [('Rate', f"@{label3}")]
    
    hover4 = HoverTool(names=['Total'])
    hover4.tooltips = [('Rate', f"@{label4}")]

    p3 = figure(
        plot_width=500,
        plot_height=450,
        title='%s per km across years'%drop_bar.value,
        x_axis_label='Year',
        y_axis_label='Count',
        tools=["box_select,save,reset",hover1,hover2,hover3,hover4]
    )
    
    p3.xaxis.ticker = years_list
    
    for i, j in zip([f+1,f+4,f+7,f+10], range(4)):
        p3.line(y=cols[i], x='Year', source=source, line_color=colors[j], line_width=4,line_alpha=0.6, legend=roads_list[j])
        p3.asterisk(y=cols[i], x='Year', source=source, color='black', name=roads_list[j], size=7)
        
    p3.legend.orientation = 'vertical'
    p3.legend.label_text_font_size = '10px'
    p3.legend.location = 'top_right'

    layout.tabs[0].child.children[0].children[1].children[1]=p3


# In[61]:


df=pd.read_csv('states_accidents.csv').transpose()
cols=list(df.loc['States/UTs'].values)
df.drop(index='States/UTs',axis=0,inplace=True)

df.columns=cols
df=df.rename_axis('Year').reset_index()
years_list=list(df['Year'].astype('int').values)

source = ColumnDataSource(df)
states_list = cols
categories=['Accidents','Persons Killed','Persons Injured']

hover = HoverTool()
hover.tooltips = [('Count', "@{Andhra Pradesh}")]

p4 = figure(
    plot_width=500,
    plot_height=450,
    title=f'Accidents across years in Andhra Pradesh',
    x_axis_label='Year',
    y_axis_label='Count',
    tools=["box_select,save,reset",hover]
)

p4.left[0].formatter.use_scientific = False
p4.xaxis.ticker = years_list
p4.line(y=cols[0], x='Year', source=source, line_color=Spectral6[0], line_width=4, line_alpha=0.6)
p4.asterisk(y=cols[0], x='Year', source=source, color='black', size=7)

drop_bar1 = Select(options=states_list, value=states_list[0])
drop_bar2 = Select(options=categories, value=categories[0])


# In[62]:


t4 = Div(text="""<b>Interpretation:</b> Number of accidents and persons injured have roughly decreased over time for most of the states and across all India. However, number of persons killed has increased over time for most of the states and overall across India.""", width=500, height=50)
c4=column(row(drop_bar1,drop_bar2),p4,t4)
show(c4)


# In[63]:


def update_bar_chart1(attrname, old, new):
    
    if drop_bar2.value==categories[0]:
        label='accidents'
        
    elif drop_bar2.value==categories[1]:
        label='killed'
        
    else:
        label='injured'
        
    df=pd.read_csv(f'states_{label}.csv').transpose()
    cols=list(df.loc['States/UTs'].values)
    df.drop(index='States/UTs',axis=0,inplace=True)
    
    df.columns=cols
    df=df.rename_axis('Year').reset_index()
    years_list=list(df['Year'].astype('int').values)
    
    source = ColumnDataSource(df)
    label = drop_bar1.value
    
    hover = HoverTool()
    hover.tooltips = [('Count', f"@{label}")]

    p4 = figure(
        plot_width=500,
        plot_height=450,
        title=f'{drop_bar2.value} across years in {drop_bar1.value}',
        x_axis_label='Year',
        y_axis_label='Count',
        tools=["box_select,save,reset",hover]
    )
    
    p4.left[0].formatter.use_scientific = False
    p4.xaxis.ticker = years_list
    p4.line(y=label, x='Year', source=source, line_color=Spectral6[0], line_width=4, line_alpha=0.6)
    p4.asterisk(y=label, x='Year', source=source, color='black', size=7)

    layout.tabs[1].child.children[0].children[1]=p4


# In[64]:


df=pd.read_csv('cities.csv')
df['accidents_per_lakh_population']=df['total_accidents']/df['population_2011']*100000
df['survival_rate']=1-df['killed']/(df['killed']+df['injured'])
df['severity_rate']=(df['killed']+df['grievously_injured'])/(df['killed']+df['injured'])

source = ColumnDataSource(df)
states_list = source.data['cities'].tolist()

# Add Tooltips
hover1 = HoverTool(names=['cities'])
hover1.tooltips = """
  <div>
    <h3>@cities</h3>
    <div><strong>Killed: </strong>@{killed}</div>
    <div><strong>Injured: </strong>@{injured}</div>
  </div>
"""

# Add plot
p81 = figure(
    plot_width=600,
    plot_height=600,
    title='Accidents across 50-million plus cities of India for year 2019',
    x_axis_label='Accidents per lakh population',
    y_axis_label='Survival probability',
    tools=["box_select,save,reset",hover1]
)

# Render glyph
p81.circle(y='survival_rate', x='accidents_per_lakh_population', 
         fill_color=linear_cmap('survival_rate', palette=Inferno256, low=0, high=1),
         source=source, radius=2.5, name='cities')

x_mean=df['accidents_per_lakh_population'].mean()
y_mean=df['survival_rate'].mean()

# Vertical line
vline = Span(location=150, dimension='height', line_color='red', line_width=2, line_alpha=0.5)
# Horizontal line
hline = Span(location=0.75, dimension='width', line_color='green', line_width=2, line_alpha=0.5)

p81.vbar(x=[75], width=150, bottom=0.3, top=0.75, color=['blue'], alpha=0.2, legend='Low Survival probability, High Accident rate')
p81.vbar(x=[75], width=150, bottom=0.75, top=1, color=['green'], alpha=0.2, legend='High Survival probability, Low Accident rate')
p81.vbar(x=[225], width=150, bottom=0.3, top=0.75, color=['red'], alpha=0.2, legend='High Accident probability, Low Survival rate')
p81.vbar(x=[225], width=150, bottom=0.75, top=1, color=['yellow'], alpha=0.2, legend='High Surival probability, High Accident rate')
p81.renderers.extend([vline, hline])

# Add Legend
p81.legend.orientation = 'vertical'
p81.legend.location = 'center_right'
p81.legend.label_text_font_size = '10px'


# In[65]:


df=pd.read_csv('cities.csv')
df['accidents_per_lakh_population']=df['total_accidents']/df['population_2011']*100000
df['survival_rate']=1-df['killed']/(df['killed']+df['injured'])
df['severity_rate']=(df['killed']+df['grievously_injured'])/(df['killed']+df['injured'])

source = ColumnDataSource(df)
states_list = source.data['cities'].tolist()

# Add Tooltips
hover1 = HoverTool(names=['cities'])
hover1.tooltips = """
  <div>
    <h3>@cities</h3>
    <div><strong>Killed: </strong>@{killed}</div>
    <div><strong>Injured: </strong>@{injured}</div>
  </div>
"""

# Add plot
p82 = figure(
    plot_width=600,
    plot_height=600,
    title='Accidents across 50-million plus cities of India for year 2019',
    x_axis_label='Accidents per lakh population',
    y_axis_label='Severity rate',
    tools=["box_select,save,reset",hover1]
)

# Render glyph
p82.circle(y='severity_rate', x='accidents_per_lakh_population', 
         fill_color=linear_cmap('severity_rate', palette=Inferno256, low=0, high=1),
         source=source, radius=2.5, name='cities')

x_mean=df['accidents_per_lakh_population'].mean()
y_mean=df['severity_rate'].mean()

# Vertical line
vline = Span(location=150, dimension='height', line_color='red', line_width=2, line_alpha=0.5)
# Horizontal line
hline = Span(location=0.75, dimension='width', line_color='green', line_width=2, line_alpha=0.5)

p82.vbar(x=[75], width=150, bottom=0, top=0.75, color=['green'], alpha=0.2, legend='Low Severity rate, High Accident rate')
p82.vbar(x=[75], width=150, bottom=0.75, top=1, color=['blue'], alpha=0.2, legend='High Severity rate, Low Accident rate')
p82.vbar(x=[225], width=150, bottom=0, top=0.75, color=['yellow'], alpha=0.2, legend='High Severity rate, Low Survival rate')
p82.vbar(x=[225], width=150, bottom=0.75, top=1, color=['red'], alpha=0.2, legend='High Severity rate, High Accident rate')
p82.renderers.extend([vline, hline])

# Add Legend
p82.legend.orientation = 'vertical'
p82.legend.location = 'center_right'
p82.legend.label_text_font_size = '10px'


# In[66]:


t81 = Div(text="""<b>Interpretation:</b> The divisions have been made corresponding to the mean rates. All the metropolitian cities lie in the green zone - which is the area of high survival probability and low accident rate.""", width=500, height=50)
t82 = Div(text="""<b>Interpretation:</b> The divisions have been made corresponding to the mean rates.""", width=500, height=50)

c81=column(p81,t81)
c82=column(p82,t82)

g2=gridplot([[c81,c82]])
show(g2)


# In[67]:


df=pd.read_csv('heatmap_india.csv')
df.rename(columns={'states':'st_nm'},inplace=True)

fp = r'india_map/india-polygon.shp'
sf_india = gpd.read_file(fp)
merged=sf_india.merge(df,on = 'st_nm', how = 'left')

#Read data to json
merged_json = json.loads(merged.to_json())

#Convert to str like object
json_data = json.dumps(merged_json)
geosource = GeoJSONDataSource(geojson = json_data)

#Define a sequential multi-hue color palette.
palette = Spectral6

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors. Input nan_color.
color_mapper = LinearColorMapper(palette = palette, low = merged['cluster_label_accident_rate'].min(),                                  high = merged['cluster_label_accident_rate'].max(), nan_color = '#d9d9d9')


#Add hover tool
hover = HoverTool(tooltips = [('State/UT','@st_nm'),('Accident_rate','@accidents_per_lakh_population'), 
                              ('Killed','@killed'),('Injured','@injured')])

#Create color bar. 
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=10, width = 400, height = 10,
                     border_line_color=None,location = (0,0), orientation = 'horizontal')

#Create figure object.
phm_1 = figure(title = 'Accidents per lakh population in India (low to high)', plot_height = 600 ,             plot_width = 450, tools = ["box_select,save,reset",hover])

phm_1.xaxis.visible = False
phm_1.yaxis.visible = False
phm_1.xgrid.grid_line_color = None
phm_1.ygrid.grid_line_color = None

#Add patch renderer to figure. 
phm_1.patches('xs','ys', source = geosource,fill_color = {'field' :'cluster_label_accident_rate', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)

#Specify layout
phm_1.add_layout(color_bar, 'below')


# In[68]:


df=pd.read_csv('heatmap_india.csv')
df.rename(columns={'states':'st_nm'},inplace=True)

fp = r'india_map/india-polygon.shp'
sf_india = gpd.read_file(fp)
merged=sf_india.merge(df,on = 'st_nm', how = 'left')

#Read data to json
merged_json = json.loads(merged.to_json())

#Convert to str like object
json_data = json.dumps(merged_json)
geosource = GeoJSONDataSource(geojson = json_data)

#Define a sequential multi-hue color palette.
palette = Spectral6[::-1]

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors. Input nan_color.
color_mapper = LinearColorMapper(palette = palette, low = merged['cluster_label_survival_rate1'].min(),                                  high = merged['cluster_label_survival_rate1'].max(), nan_color = '#d9d9d9')

#Add hover tool
hover = HoverTool(tooltips = [('State/UT','@st_nm'),('Survival Rate','@survival_rate'),
                              ('Killed','@killed'),('Injured','@injured')])

#Create color bar. 
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=5, width = 400, height = 10,
                     border_line_color=None,location = (-5,0), orientation = 'horizontal')

#Create figure object.
phm_2 = figure(title = 'Survival Rate in India (low to high)',            plot_height = 600 , plot_width = 450, tools = ["box_select,save,reset",hover])

phm_2.xaxis.visible = False
phm_2.yaxis.visible = False
phm_2.xgrid.grid_line_color = None
phm_2.ygrid.grid_line_color = None

#Add patch renderer to figure. 
phm_2.patches('xs','ys', source = geosource,fill_color = {'field' :'cluster_label_survival_rate1', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)

#Specify layout
phm_2.add_layout(color_bar, 'below')


# In[69]:


df=pd.read_csv('heatmap_india.csv')
df.rename(columns={'states':'st_nm'},inplace=True)

fp = r'india_map/india-polygon.shp'
sf_india = gpd.read_file(fp)
merged=sf_india.merge(df,on = 'st_nm', how = 'left')

#Read data to json
merged_json = json.loads(merged.to_json())

#Convert to str like object
json_data = json.dumps(merged_json)
geosource = GeoJSONDataSource(geojson = json_data)

#Define a sequential multi-hue color palette.
palette = Spectral6

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors. Input nan_color.
color_mapper = LinearColorMapper(palette = palette, low = merged['cluster_label_fatality_rate'].min(),                                  high = merged['cluster_label_fatality_rate'].max(), nan_color = '#d9d9d9')

#Add hover tool
hover = HoverTool(tooltips = [('State/UT','@st_nm'),('Severity Rate','@severity_rate'),
                              ('Killed','@killed'),('Injured','@injured')])

#Create color bar. 
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=5, width = 400, height = 10,
                     border_line_color=None,location = (-5,0), orientation = 'horizontal')

#Create figure object.
phm_3 = figure(title = 'Severity of accidents (low to high)',            plot_height = 600 , plot_width = 450, tools = ["box_select,save,reset",hover])

phm_3.xaxis.visible = False
phm_3.yaxis.visible = False
phm_3.xgrid.grid_line_color = None
phm_3.ygrid.grid_line_color = None

#Add patch renderer to figure. 
phm_3.patches('xs','ys', source = geosource,fill_color = {'field' :'cluster_label_fatality_rate', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)

#Specify layout
phm_3.add_layout(color_bar, 'below')


# In[70]:


thm = Div(text="""<b>Interpretation:</b> There are some states which have relatively lower accident rates while having relatively higher severity rates. For example: Uttar Pradesh, Bihar, Jharkhand, West Bengal. On the other hand, there are some states which have a relatively higher accident rates while having lower severity rates. For example: Tamil Nadu, Telangana, Madhya Pradesh. Also, there are some states which have a relatively lower accident rates while having relatively higher survival rates. For example: Rajasthan, Gujarat, Maharashtra, Odisha. """, width=1400, height=50)
c79=column(row(phm_1,phm_2,phm_3),thm)
show(c79)


# In[71]:


df=pd.read_csv('vehicle.csv')
df['accidents per 1000 vehicles per 1000 km']=df["Road Accidents ('000)"]/(df['Road Length (000 km) ']*df["Registered Vehicles ('000)"])*1000
df['Vehicle density (no. of vehicles per km of road)']*=1000

df.rename(columns={'Vehicle density (no. of vehicles per km of road)':                   'Vehicle density (no. of vehicles per 1000 km of road)'}, inplace=True)
          
df.dropna(inplace=True)
cols=list(df.columns)

x=(df[cols[-2]].values)
y=(df[cols[-1]].values)
order = np.argsort(x)

df_lowess=pd.read_csv('lowess_results.csv')
y_sm=df_lowess['y_sm'].values
y_std=df_lowess['y_std'].values

source = ColumnDataSource(df)

# Add plot
p6 = figure(
    plot_width=500,
    plot_height=500,
    title='Lowess Curve',
    x_axis_label='Vehicles per 1000 km',
    y_axis_label='Accidents per 1000 vehicles per 1000 km',
    tools="box_select,save,reset"
)

# Render glyph
p6.line(y=y_sm[order], x=x[order], line_color='red', line_width=4, line_alpha=0.4, legend='LOWESS')
p6.circle(y=cols[-1], x=cols[-2], source=source, color='black', legend='Observations', name='observations')
p6.varea(x[order], y_sm[order] - 1.96*y_std[order],
                 y_sm[order] + 1.96*y_std[order], alpha=0.2, legend='LOWESS uncertainty')
# Add Legend
p6.legend.orientation = 'vertical'
p6.legend.location = 'top_right'
p6.legend.label_text_font_size = '10px'

hover = HoverTool(names=['observations'])
hover.tooltips = [('Year','@Year')]

p6.add_tools(hover)


# In[72]:


t6 = Div(text="""<b>Interpretation:</b> Vehicles per 1000 km is an increasing function of time. However, if the vehicle density increases, the accident rate does not increase, the accident rate has practically flattened out. This implies that drivers/passengers might not be the reason for such accidents.""", width=500, height=70)
c6=column(p6,t6)
show(c6)


# In[93]:


from bokeh.models import FactorRange
df=pd.read_csv('roads_share.csv')
cols=list(df.columns)
cols.remove('Year')

titles=['accidents','persons killed','persons injured']
for f in range(3):

    iter_df=df.loc[:,['Year',cols[f+0],cols[f+3],cols[f+6]]]

    years=list(iter_df['Year'].astype('str').values)
    roads=['NH', 'SH', 'OR']

    x = [(year, road) for year in years for road in roads]

    iter_cols=list(iter_df.columns)
    iter_cols.remove('Year')

    counts = iter_df.loc[:,iter_cols].values.reshape(-1)

    source = ColumnDataSource(data=dict(x=x, counts=counts))

    p = figure(x_range=FactorRange(*x), plot_height=350, plot_width=650, title=f"% share of {titles[f]} by Year")

    p.vbar(x='x', top='counts', width=0.9, source=source,
                 fill_color=factor_cmap('x', palette=Accent[3], factors=roads, start=1, end=2))

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None

    vars()[f'vbar{f+1}']=p
    
p11=gridplot([[vbar1,None],[vbar2,vbar3]])


# In[94]:


t11 = Div(text="""<b>Interpretation:</b> The percentage share of accidents, persons killed and persons injured have roughly remained the same across roads in the last twelve years.""", width=1000, height=70)
c10=column(p11,t11)
show(c10)


# In[26]:


drop_bar.on_change("value", update_bar_chart)
drop_bar1.on_change("value", update_bar_chart1)
drop_bar2.on_change("value", update_bar_chart1)


# In[84]:


first=Panel(child=column(row(c1,c3),c10),title='Roads')
second=Panel(child=column(c4,g1),title='States')
third=Panel(child=column(g2),title='Million Plus cities')
fourth=Panel(child=row(c6),title='Vehicle Density')
fifth=Panel(child=row(c79),title='INDIA Heatmap')

layout=Tabs(tabs=[first,second,third,fourth,fifth])
curdoc().add_root(layout)


# In[ ]:




