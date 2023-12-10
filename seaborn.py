#!/usr/bin/env python
# coding: utf-8

# # Seaborn 

# ### Import libraries

# In[8]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ### Read dataset
# 
# In this kernel, I will focus on those datasets which help to explain various features of Seaborn.  So, I will read the related datasets with pandas read_csv() function.

# In[11]:


fifa19 = pd.read_csv(r'D:\NIT\2 DEC\24th- Seaborn, Eda practicle\Seaborn\FIFA.csv')


# ### Exploratory Data Analysis

# ### Preview the dataset

# In[12]:


fifa19.head()


# ### View summary of dataset

# In[13]:


fifa19.info()


# In[14]:


fifa19['Body Type'].value_counts()


# #### Comment
# 
# 
# - This dataset contains 89 variables.
# 
# - Out of the 89 variables, 44 are numerical variables. 38 are of float64 data type and remaining 6 are of int64 data type.
# 
# - The remaining 45 variables are of character data type.
# 
# - Let's explore this further.
# 
# 

# ### Explore `Age` variable

# ### Visualize distribution of `Age` variable with Seaborn `distplot()` function
# 
# - Seaborn `distplot()` function flexibly plots a univariate distribution of observations.
# 
# - This function combines the matplotlib hist function (with automatic calculation of a good default bin size) with the seaborn kdeplot() and rugplot() functions.
# 
# * - So, let's visualize the distribution of `Age` variable with Seaborn `distplot()` function.

# # Comment
# It can be seen that the Age variable is slightly positively skewed.
# We can use Pandas series object to get an informative axis label as follows-
# 
# We can plot the distribution on the vertical axis as follows:-

# In[15]:


ax=plt.subplots(figsize=(8,6))
ax=sns.distplot(fifa19['Age'])


# In[16]:


ax=plt.subplots(figsize=(8,6))
ax=sns.distplot(fifa19['Age'], bins=10)


# In[17]:


fifa19['Age'].max()


# In[18]:


ax=plt.subplots(figsize=(8,6))
x=fifa19['Age']
x=pd.Series(x, name="Age variable")
ax=sns.distplot(x, bins=10)

# We can plot the distribution on the vertical axis as follows:-
# In[21]:


ax=plt.subplots(figsize=(8,6))
ax=sns.distplot(fifa19['Age'], bins=10,vertical=True)


# ### Seaborn Kernel Density Estimation (KDE) Plot
# 
# 
# - The `kernel density estimate (KDE)` plot is a useful tool for plotting the shape of a distribution. 
# 
# - Seaborn kdeplot is another seaborn plotting function that fits and plot a univariate or bivariate kernel density estimate.
# 
# - Like the histogram, the KDE plots encode the density of observations on one axis with height along the other axis.
# 
# - We can plot a KDE plot as follows-

# In[22]:


ax=plt.subplots(figsize=(8,6))
ax=sns.kdeplot(fifa19['Age'] )


# In[23]:


f, ax = plt.subplots(figsize=(8,6))
x = fifa19['Age']
x = pd.Series(x, name="Age variable")
ax = sns.kdeplot(x)


# We can shade under the density curve and use a different color as follows:-

# In[25]:


ax=plt.subplots(figsize=(8,6))
ax=sns.kdeplot(fifa19['Age'] ,color="r" ,shade=True)


# ### Histograms
# 
# - A histogram represents the distribution of data by forming bins along the range of the data and then drawing bars to show the number of observations that fall in each bin. 
# 
# - A `hist()` function already exists in matplotlib. 
# 
# - We can use Seaborn to plot a histogram.
# 

# In[26]:


ax=plt.subplots(figsize=(8,6))
ax=sns.distplot(fifa19['Age'],kde=False)


# In[27]:


ax=plt.subplots(figsize=(8,6))
ax=sns.distplot(fifa19['Age'],kde=False,bins=12,rug=True)


# We can plot a KDE plot alternatively as follows:-

# In[28]:


f, ax = plt.subplots(figsize=(8,6))
x = fifa19['Age']
ax = sns.distplot(x, hist=False, rug=True, bins=10)
plt.show()


# ### Explore `Preferred Foot` variable

# ### Check number of unique values in `Preferred Foot` variable

# In[29]:


fifa19['Preferred Foot'].nunique()


# We can see that there are two types of unique values in `Preferred Foot` variable.

# ### Check frequency distribution of values in `Preferred Foot` variable

# In[30]:


fifa19['Preferred Foot'].value_counts()


# The `Preferred Foot` variable contains two types of values - `Right` and `Left`.

# ### Visualize distribution of values with Seaborn `countplot()` function.
# 
# - A countplot shows the counts of observations in each categorical bin using bars.
# 
# - It can be thought of as a histogram across a categorical, instead of quantitative, variable.
# 
# - This function always treats one of the variables as categorical and draws data at ordinal positions (0, 1, … n) on the relevant axis, even when the data has a numeric or date type.
# 
# 1. - We can visualize the distribution of values with Seaborn `countplot()` function as follows-

# In[31]:


f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="Preferred Foot", data=fifa19)


# We can show value counts for two categorical variables as follows-

# In[32]:


f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="Preferred Foot", data=fifa19,hue="Preferred Foot")


# We can draw plot vertically as follows-

# In[33]:


f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(y="Preferred Foot", data=fifa19,color="c")


# In[34]:


fifa19["Real Face"]


# In[35]:


fifa19['Real Face'].value_counts()


# In[36]:


f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="Preferred Foot", hue="Real Face", data=fifa19)


# ### Seaborn `Catplot()` function
# 
# - We can use Seaborn `Catplot()` function to plot categorical scatterplots.
# 
# - The default representation of the data in `catplot()` uses a scatterplot. 
# 
# - It helps to draw figure-level interface for drawing categorical plots onto a facetGrid.
# 
# - This function provides access to several axes-level functions that show the relationship between a numerical and one or more categorical variables using one of several visual representations. 
# 
# - The `kind` parameter selects the underlying axes-level function to use.

# We can use the kind parameter to draw different plot kin to visualize the same data. We can use the Seaborn `catplot()` function to draw a `countplot()` as follows-

# In[37]:


g = sns.catplot(data=fifa19,x="Preferred Foot", kind="count", palette="ch:.25")


# In[38]:


g = sns.catplot(data=fifa19,x="Preferred Foot", kind="count", palette="muted")


# ### Explore `International Reputation` variable

# ### Check the number of unique values in `International Reputation` variable

# In[39]:


fifa19['International Reputation'].nunique()


# ### Check the distribution of values in `International Reputation` variable

# In[40]:


fifa19['International Reputation'].value_counts()


# ### Seaborn `Stripplot()` function
# 
# 
# - This function draws a scatterplot where one variable is categorical.
# 
# - A strip plot can be drawn on its own, but it is also a good complement to a box or violin plot in cases where we want to show all observations along with some representation of the underlying distribution.
# 
# - I will plot a stripplot with `International Reputation` as categorical variable and `Potential` as the other variable.

# In[41]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", data=fifa19,hue="Potential")


# In[42]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", data=fifa19,hue="International Reputation")


# We can add jitter to bring out the distribution of values as follows-

# In[43]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", data=fifa19, jitter=False,hue="Potential")


# In[46]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot", 
                   data=fifa19)


# We can nest the strips within a second categorical variable - `Preferred Foot` as folows-

# In[47]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot", 
                   data=fifa19,palette="Set2",dodge=True)


# In[48]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot", 
                   data=fifa19, palette="Set2", size=10)


# In[49]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot", 
                   data=fifa19, palette="Set2", size=20,marker="d",edgecolor="pink")


# We can draw strips with large points and different aesthetics as follows-

# In[50]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot", 
                   data=fifa19, palette="Set2", size=20, marker="D",
                   edgecolor="gray", alpha=.12)


# ### Seaborn `boxplot()` function
# 
# 
# - This function draws a box plot to show distributions with respect to categories.
# 
# - A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables or across levels of a categorical variable. 
# 
# - The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, except for points that are determined to be “outliers” using a method that is a function of the inter-quartile range.
# 
# - I will plot the boxplot of the `Potential` variable as follows-

# In[51]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=fifa19["Potential"])


# We can draw the vertical boxplot grouped by the categorical variable `International Reputation` as follows-

# In[52]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="International Reputation", y="Potential", data=fifa19,hue="International Reputation")


# We can draw a boxplot with nested grouping by two categorical variables as follows-

# In[53]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="International Reputation", y="Potential", data=fifa19,hue="International Reputation")


# ### Seaborn `violinplot()` function
# 
# 
# - This function draws a combination of boxplot and kernel density estimate.
# 
# - A violin plot plays a similar role as a box and whisker plot. 
# 
# - It shows the distribution of quantitative data across several levels of one (or more) categorical variables such that those distributions can be compared. 
# 
# - Unlike a box plot, in which all of the plot components correspond to actual datapoints, the violin plot features a kernel density estimation of the underlying distribution.
# 
# - I will plot the violinplot of `Potential` variable as follows-

# In[54]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x=fifa19["Potential"])


# We can draw the vertical violinplot grouped by the categorical variable `International Reputation` as follows-

# In[55]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x="International Reputation", y="Potential", data=fifa19)


# In[56]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x="International Reputation", y="Potential", data=fifa19,hue="International Reputation")


# We can draw a violinplot with nested grouping by two categorical variables as follows-

# In[57]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19, palette="muted")


# In[58]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x="International Reputation", y="Potential", hue="Preferred Foot", 
               data=fifa19, palette="muted")


# We can draw split violins to compare the across the hue variable as follows-

# In[59]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x="International Reputation", y="Potential", hue="Preferred Foot", 
               data=fifa19, palette="muted", split=True)


# ### Seaborn `pointplot()` function
# 
# 
# - This function show point estimates and confidence intervals using scatter plot glyphs.
# 
# - A point plot represents an estimate of central tendency for a numeric variable by the position of scatter plot points and provides some indication of the uncertainty around that estimate using error bars.

# In[60]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential", data=fifa19)


# We can draw a set of vertical points with nested grouping by a two variables as follows-

# In[61]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19


# We can separate the points for different hue levels along the categorical axis as follows-

# In[62]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19)


# In[63]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", 
              data=fifa19, markers=["p", "x"])


# We can use a different marker and line style for the hue levels as follows-

# In[64]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", 
              data=fifa19, markers=["o", "x"], linestyles=["-.", "--"])


# ### Seaborn `barplot()` function
# 
# 
# - This function show point estimates and confidence intervals as rectangular bars.
# 
# - A bar plot represents an estimate of central tendency for a numeric variable with the height of each rectangle and provides some indication of the uncertainty around that estimate using error bars. 
# 
# - Bar plots include 0 in the quantitative axis range, and they are a good choice when 0 is a meaningful value for the quantitative variable, and you want to make comparisons against it.
# 
# - We can plot a barplot as follows-

# In[65]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa19)


# In[66]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa19,hue="International Reputation")


# We can draw a set of vertical bars with nested grouping by a two variables as follows-

# In[103]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19)


# We can use median as the estimate of central tendency as follows-

# In[104]:


from numpy import median
f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa19, estimator=median)
plt.show()


# In[105]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa19, ci=68)
plt.show()


# We can show standard deviation of observations instead of a confidence interval as follows-

# In[106]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa19, ci="sd")
plt.show()


# We can add “caps” to the error bars as follows-

# In[107]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa19, capsize=0.2)
plt.show()


# ### Visualizing statistical relationship with Seaborn `relplot()` function

# ### Seaborn `relplot()` function
# 
# 
# - Seaborn `relplot()` function helps us to draw figure-level interface for drawing relational plots onto a FacetGrid.
# 
# - This function provides access to several different axes-level functions that show the relationship between two variables with semantic mappings of subsets. 
# 
# - The `kind` parameter selects the underlying axes-level function to use-
# 
# - scatterplot() (with kind="scatter"; the default)
# 
# - lineplot() (with kind="line")

# We can plot a scatterplot with variables `Heigh` and `Weight` with Seaborn `relplot()` function as follows-

# In[70]:


g = sns.relplot(x="Overall", y="Potential", data=fifa19)


# ### Seaborn `scatterplot()` function
# 
# 
# - This function draws a scatter plot with possibility of several semantic groups.
# 
# - The relationship between x and y can be shown for different subsets of the data using the `hue`, `size` and `style` parameters. 
# 
# - These parameters control what visual semantics are used to identify the different subsets.

# In[71]:


f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="Height", y="Weight", data=fifa19)
plt.show()


# ### Seaborn `lineplot()` function
# 
# 
# - THis function draws a line plot with possibility of several semantic groupings.
# 
# - The relationship between x and y can be shown for different subsets of the data using the `hue`, `size` and `style` parameters. 
# 
# - These parameters control what visual semantics are used to identify the different subsets.

# In[72]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.lineplot(x="Stamina", y="Strength", data=fifa19)
plt.show()


# ### Visualize linear relationship with Seaborn `regplot()` function

# ### Seaborn `regplot()` function
# 
# - This function plots data and a linear regression model fit.
# 
# - We can plot a linear regression model between `Overall` and `Potential` variable with `regplot()` function as follows-

# In[73]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="Overall", y="Potential", data=fifa19)
plt.show()


# We can use a different color and marker as follows-

# In[74]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="Overall", y="Potential", data=fifa19, color= "g", marker="+")
plt.show()


# We can plot with a discrete variable and add some jitter as follows-

# In[75]:


f, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x="International Reputation", y="Potential", data=fifa19, x_jitter=.01)
plt.show()


# ### Seaborn `lmplot()` function
# 
# 
# - This function plots data and regression model fits across a FacetGrid.
# 
# - This function combines `regplot()` and `FacetGrid`. 
# 
# - It is intended as a convenient interface to fit regression models across conditional subsets of a dataset.
# 
# - We can plot a linear regression model between `Overall` and `Potential` variable with `lmplot()` function as follows-

# In[76]:


g= sns.lmplot(x="Overall", y="Potential", data=fifa19)


# We can condition on a third variable and plot the levels in different colors as follows-

# In[77]:


g= sns.lmplot(x="Overall", y="Potential", hue="Preferred Foot", data=fifa19)


# We can use a different color palette as follows-

# In[78]:


g= sns.lmplot(x="Overall", y="Potential", hue="Preferred Foot", data=fifa19, palette="Set1")


# We can plot the levels of the third variable across different columns as follows-

# In[79]:


g= sns.lmplot(x="Overall", y="Potential", col="Preferred Foot", data=fifa19)


# ### Multi-plot grids

# ### Seaborn `FacetGrid()` function
# 
# - The FacetGrid class is useful when you want to visualize the distribution of a variable or the relationship between multiple variables separately within subsets of your dataset. 
# 
# - A FacetGrid can be drawn with up to three dimensions - `row`, `col` and `hue`. The first two have obvious correspondence with the resulting array of axes - the `hue` variable is a third dimension along a depth axis, where different levels are plotted with different colors.
# 
# - The class is used by initializing a FacetGrid object with a dataframe and the names of the variables that will form the `row`, `column` or `hue` dimensions of the grid. 
# 
# - These variables should be categorical or discrete, and then the data at each level of the variable will be used for a facet along that axis.
# 

# We can initialize a 1x2 grid of facets using the fifa19 dataset.

# In[80]:


g = sns.FacetGrid(fifa19, col="Preferred Foot")


# We can draw a univariate plot of `Potential` variable on each facet as follows-

# In[81]:


g = sns.FacetGrid(fifa19, col="Preferred Foot")
g = g.map(plt.hist, "Potential")


# In[82]:


g = sns.FacetGrid(fifa19, col="Preferred Foot")
g = g.map(plt.hist, "Potential", bins=10, color="r")


# We can plot a bivariate function on each facet as follows-

# In[83]:


g = sns.FacetGrid(fifa19, col="Preferred Foot")
g = (g.map(plt.scatter, "Height", "Weight", edgecolor="w").add_legend())


# The size of the figure is set by providing the height of each facet, along with the aspect ratio:

# In[84]:


g = sns.FacetGrid(fifa19, col="Preferred Foot", height=5, aspect=1)
g = g.map(plt.hist, "Potential")


# ### Seaborn `Pairgrid()` function
# 
# 
# - This function plots subplot grid for plotting pairwise relationships in a dataset.
# 
# - This class maps each variable in a dataset onto a column and row in a grid of multiple axes. 
# 
# - Different axes-level plotting functions can be used to draw bivariate plots in the upper and lower triangles, and the the marginal distribution of each variable can be shown on the diagonal.
# 
# - It can also represent an additional level of conditionalization with the hue parameter, which plots different subets of data in different colors. 
# 
# - This uses color to resolve elements on a third dimension, but only draws subsets on top of each other and will not tailor the hue parameter for the specific visualization the way that axes-level functions that accept hue will.

# In[85]:


fifa19_new = fifa19[['Age', 'Potential', 'Strength', 'Stamina', 'Preferred Foot']]


# In[86]:


g = sns.PairGrid(fifa19_new)
g = g.map(plt.scatter)


# We can show a univariate distribution on the diagonal as follows-

# In[87]:


g = sns.PairGrid(fifa19_new)
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)


# We can color the points using the categorical variable `Preferred Foot` as follows -

# In[88]:


g = sns.PairGrid(fifa19_new, hue="Preferred Foot")
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()


# We can use a different style to show multiple histograms as follows-

# In[89]:


g = sns.PairGrid(fifa19_new, hue="Preferred Foot")
g = g.map_diag(plt.hist, histtype="step", linewidth=3)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()


# We can plot a subset of variables as follows-

# In[90]:


g = sns.PairGrid(fifa19_new, vars=['Age', 'Stamina'])
g = g.map(plt.scatter)


# ### Seaborn `Jointgrid()` function
# 
# 
# - This function provides a grid for drawing a bivariate plot with marginal univariate plots.
# 
# - It set up the grid of subplots.

# We can initialize the figure and add plots using default parameters as follows-

# In[91]:


g = sns.JointGrid(x="Overall", y="Potential", data=fifa19)
g = g.plot(sns.regplot, sns.distplot)


# We can draw the join and marginal plots separately, which allows finer-level control other parameters as follows -

# In[92]:


import matplotlib.pyplot as plt


# In[93]:


g = sns.JointGrid(x="Overall", y="Potential", data=fifa19)
g = g.plot_joint(plt.scatter, color=".5", edgecolor="white")
g = g.plot_marginals(sns.distplot, kde=False, color=".5")


# In[94]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="Overall", y="Potential", data=fifa19);


# In contrast, the size and shape of the `lmplot()` figure is controlled through the FacetGrid interface using the size and aspect parameters, which apply to each facet in the plot, not to the overall figure itself.

# In[95]:


sns.lmplot(x="Overall", y="Potential", col="Preferred Foot", data=fifa19, col_wrap=2, height=5, aspect=1)


# ### Seaborn figure styles
# 
# 
# - There are five preset seaborn themes: `darkgrid`, `whitegrid`, `dark`, `white` and `ticks`. 
# 
# - They are each suited to different applications and personal preferences. 
# 
# - The default theme is darkgrid. 
# 
# - The grid helps the plot serve as a lookup table for quantitative information, and the white-on grey helps to keep the grid from competing with lines that represent data. 
# 
# - The whitegrid theme is similar, but it is better suited to plots with heavy data elements:
# 
# 

# I will define a simple function to plot some offset sine waves, which will help us see the different stylistic parameters as follows -

# In[96]:


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)


# This is what the plot looks like with matplotlib default parameters.

# In[97]:


sinplot()


# To switch to seaborn defaults, we need to call the `set()` function as follows -

# In[98]:


sns.set()
sinplot()


# - We can set different styles as follows -

# In[99]:


sns.set_style("whitegrid")
sinplot()


# In[100]:


sns.set_style("dark")
sinplot()


# In[101]:


sns.set_style("white")
sinplot()


# In[102]:


sns.set_style("ticks")
sinplot()

