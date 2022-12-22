#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-warning"> <b>Team Lead comment:</b> Hi! thanks for project! Actually, I love how you technically approached the project! I'm sure that you will "polish it" a bit and add more structured conclusions and comments! There are some of my comments below as well ;) besides that, you did a great job! good luck! 
#    
# </div>

# https://drive.google.com/file/d/10zGT2Da13gpRExwRFou1u9o6d3wecH9b/view?usp=sharing

# # <center>Research on E-commerse product range<center>
# 
# 
# We are making an analysis for the **e-commerse company**. The company offers its clients a huge range of products. Using the data of allmost the year sells , we need to find out the specifity of the product range in the company.\
# We have dataset, that contains the information about the item cost, the quantity of products ordered, the description of the product and the number of invoice and client. The goal is to **analyse the product range**, the categories of the items we sell and **to understand the most profitable categories** in our store.\
# Let's move on and study the data more thorougly. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
from scipy import stats
import plotly.express as px
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.ticker as ticker


# In[2]:


try:
    df = pd.read_csv('ecommerce_dataset_us.csv', sep = '\t')
except:
    df = pd.read_csv('/datasets/ecommerce_dataset_us.csv', sep = '\t')


# ## Exploratory Data Analysis.

# In[3]:


df.head()


# In[4]:


df.tail()


# The data contains the customerId, the number of Invoice, the description of product, the ordered quantity, the  invoice date and the price of each item. To make the analysis more convenient it is recommended to write the columns names and description in lower letters, let's do that

# In[5]:


df.columns= df.columns.str.lower()


# In[6]:


df.info()


# There is ~541 thousand rows in our data, allmost all columns have the full data, only description and customerid have missing values. We see that the quantity has an integer format, but the unitprice and customerid has float format (which is better to change to int) and the date of invoice is object (recommended to change to the date format).

# In[7]:


df.describe(include='all')


# The table above can tell us a lot of general information about the data we have. We see that there are 25900 unique invoices and the 4057 stockcodes, the mean unitprice is 4.61, we also see the negative unitprices and that the maximum price is 38970. Let's  start with analyse the missing values (find the % of missing values)

# In[8]:


df_null = df.isnull().sum()


# In[9]:


df_null / len(df) * 100


# In[10]:


for i in df:
    if df[i].isnull().sum() > 0:
        print(i)


# We see that the % of missing values in description is very little, so we can remove it, and as the goal is to analyse the product range and not the customer behaviour, we can't just remove the rows with the upsent customerid, so we will replace it with "0" value, to keep as many rows with the items we sell as possible.

# In[11]:


df['customerid'] = df['customerid'].fillna(0).astype(int)


# In[12]:


df = df.dropna(subset=[2], axis=1)


# In[13]:


df[df['description'].isnull()]


# The description is written in capital letters, it is time to fix it.

# In[14]:


df['description']=df['description'].str.lower()
df['description'].head()


# In[15]:


df['description'].unique()


# In[16]:


df[df['description']=='lost']


# Starting the dates analysis, we point out that the the data starts at 29.11.2018 and ends in Decemver 2019. The first date is 29.11 and the last date 07.12. It is better to have the whole month for the investigation,so i ll remove one day of november and a week of december 2019.

# In[17]:


df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')


# In[18]:


df['invoicedate'].min()


# In[19]:


df['invoicedate'].max()


# In[20]:


df = df[(df['invoicedate'] >= '2018-12-01') & (df['invoicedate'] < '2019-12-01')]


# In[21]:


print(df['invoicedate'].min())
print(df['invoicedate'].max())


# It is time to make a little research about the unitprice. We have the negative prices, let's figure out what are they for.

# In[22]:


df['unitprice'].min()


# In[23]:


df[df['unitprice']<0]


# The point is that the negative unitpraice stands for the bad debt. It is more relevant for Accounting and has no meaning for the product range analysis.

# In[24]:


df['unitprice'].describe()


# In the raw dataset we also don't have a column that shows us the total revenue. So we need to make one - we will name it as "total_inv"

# In[25]:


df['total_inv'] = df['unitprice'] * df['quantity']
df.head()


# In[26]:


df['total_inv'].describe()


# In[27]:


rows_negative = df[df['total_inv']<0]
rows_negative


# Now we see that there are ~9.2 thousands of negative total invoice rows. From the first look we see that there are some returns or accounting records. We don't need them for the future investigation, so we will work with just the positive ones.

# In[28]:


df_positive= df[df['total_inv']>0]
df_positive.info()


# In[29]:


df_positive['total_inv']=df_positive['total_inv'].astype(int)


# In[30]:


df_positive.duplicated().value_counts()


# Also there are some duplicated rows in the dataset (4923 rows), the share of duplicated rows is very small so to have the cleaner data we can remove them either.

# In[31]:


df_positive.duplicated().sum() / len(df) *100


# In[32]:


df_positive.drop_duplicates(inplace=True)


# In[33]:


df_positive.duplicated().value_counts()


# After cleaning the data and removing the missing values, the duplicated rows abd negative values ( containing accounting rows and returns) we have tha data that is suitable for analysis

# In[34]:


df_positive['total_inv'].median()


# In[35]:


df_positive['date']=df_positive['invoicedate'].dt.strftime('%Y-%m-%d')


# In[36]:


df_positive['month'] = df_positive['invoicedate'].dt.strftime('%Y-%m')


# In[37]:


df_positive.info()


# In[38]:


df_positive.head()


# Having the clean data, we can move to the next step to investigate if we have some outliers in our dataset. We are going to analyse the abnormal total invoice values.

# In[39]:


def find_outliers_IQR(df_positive):

   q1=df_positive.quantile(0.25)

   q3=df_positive.quantile(0.75)

   IQR=q3-q1

   outliers = df_positive[((df_positive<(q1-1.5*IQR)) | (df_positive>(q3+1.5*IQR)))]

   return outliers


# In[40]:


outliers = find_outliers_IQR(df_positive['total_inv'])

print('number of outliers: ' + str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers.head()


# We do have outliers, but we will consider them as the normal part of our dataset, as the client can have the another behaviour comparing to the other clients, but also can give us a really usefull information about our product. It is also very important for us to understand the price segment that our company represents. So we are going to categorize the items in terms of price information. Do we have items which reffer to economy segment, are there any premium segment products?

# In[41]:


def price_range(row):
    if row['unitprice']> 0 and row['unitprice'] < 500:
        return 'economy segment'
    if row['unitprice']>500 and row['unitprice'] < 3000:
        return "medium low"
    if row['unitprice']>3000 and row['unitprice'] < 9000:
        return 'medium high'
    if row['unitprice']>9000 and row['unitprice']<35000:
        return 'expensive'
    if row['unitprice']>35000:
        return ['premium']
    else:
        return ['not_price']


# In[42]:


df_positive = df_positive[~df_positive['description'].isin(['manual', 'amazon fee', 'samples', 'adjust bad debt', 'lost'])]


# In[43]:


df_positive['price_segment']=df_positive.apply(price_range, axis = 1)


# In[44]:


df_positive[df_positive['price_segment']=='not_price']


# In[45]:


df_positive.describe(include='all',datetime_is_numeric=True)


# In[46]:


price_df =  pd.pivot_table(df_positive, values='total_inv', index=['month'], columns = ['price_segment'], aggfunc='sum',fill_value=0)
price_df


# So after all we notice that the huge amount of our products refers to economy segment. Very few products belong to  medium low and medium high pricing category. We can conclude that our company belongs to the companies selling goods of a low price category and don't have any unique and expensive items.

# In[47]:


avg_check=df_positive.groupby(['month','invoiceno'])['total_inv'].sum().reset_index()
avg_check.head().sort_values(by='total_inv')


# In[48]:


average_purchase_size = df_positive['total_inv'].mean()
average_purchase_size


# The average purchase size that was made in our company cost 19.30. The next step to analyse the revenue per month 

# In[49]:


grouped_revenue = df_positive.groupby(['month'])['total_inv'].sum()
grouped_revenue


# As we look at the table above, we can describe the 11.2019  as the absolute success, and we see that the revenue is growing from month to month, the exception are Feb. and Apr. 2019 where we see the fall of the total revenue. We definetly should check what happened with sales on that dates.

# In[50]:


grouped_checks = df_positive.groupby(['month'])['invoiceno'].nunique()
grouped_checks


# The situation with orders is the same, the number of orders is stabily growing, we started from 1285 orders in December 2018 and in November 2019 we see  that there are already 2840 orders. Also after cleaning the data there are actually 3908 unique items left ( we deleted all irrelevant descriptions)

# In[51]:


grouped_checks.mean()


# In[52]:


df_positive['stockcode'].nunique()


# Top-selling products aren’t necessarily the most profitable. When determining which products should occupy the most of the company, let's take a look at what product sold the most now? How many items were ordered and what is the unitprice of the top 5 products.

# In[53]:


product_group = df_positive.groupby(['stockcode','description'])
pr_gr = product_group.sum().sort_values('quantity', axis=0, ascending=False).head(5)
pr_gr.reset_index(inplace=True)
pr_gr


# Now we notice that the leader is medium ceramic top storage jar, 77856 items were sold and the sale of this product brought us a total profit equall to 81412 USD, on the 2nd place we see the world war 2 gliders asstd designs with the total sales of 50614 items and total invoice 12490. The thing is really important is that the white hanging heart t-light holder	which is only on the 4th place in order of number of items sold brought us the most total revenue which is equall to 99196 USD. 

# In[54]:


fig2 = px.bar(pr_gr, x=pr_gr['description'] , y=pr_gr['quantity'],color='description', range_color=[5,8],title="Top 5 sold items")
fig2.update_xaxes(tickangle=45)
fig2.update_layout(
    title={
        'text': "Top 5 sold items",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)
fig2.show() 


# In[55]:


contains_non_string = df_positive[df_positive['description'].apply(type) != str].any()
contains_non_string


# In[56]:


df_positive['description']=df_positive['description'].astype(str)


# <div class="alert alert-success"> <b>Team Lead comment:</b> +
# </div>

# ## Product Categorization and Category Analysis

# The very important but not the last step is to find out what type of products do we sell? Do we have a big mix of products or the product line is narrow. Which categories are sold more and bring more money for the company? To define the categories I'll use lemmatization function.

# In[57]:


wordnet_lemma = WordNetLemmatizer()


# In[58]:


def lemmatize_text(text):
    return [wordnet_lemma.lemmatize(w) for w in nltk.word_tokenize(text)]


# In[59]:


dataset = df_positive
col = ['description']
dataset= dataset[col]
dataset.dropna(subset=["description"], inplace=True)


# In[60]:


dataset = dataset.sample(frac=1)
dataset.head()


# In[61]:


products_lemmatized = dataset['description'].apply(lemmatize_text)


# In[62]:


houseware_category = ['station','triobase','fragrance','electronic','dovecote','planter','key-chains','keyring','ashtray','candlestick','lightbulb','burner','seat','shelf','doorknob','image','placemat','mirror','list','pouffe','base','chime','mushroom','shell','incense/cand','squarecushion','paperweight','incense/candle','washbag','swat','flask','level','memoboard','tape', 'chandelier','measure','birdhouse','photoframe','cabinet', 'cage','incense+flower','hand', 'headphone','basin','screwdriver','clay','tree','toadstool','feeder','drawerknob','torch','incense/candl','peg','crate','number','design', 'mat','curtain','t-light','t-lights', 'washing','chocolatecandle','lampshade','case','calendar','stand','can','hook','tin','hammock','tray','gardener','hanging','chalkboard','tumbler','rose','windmill','pegs','measuring','incense','bathroom','container','tlights','garden','spade','housework','stool','flower','radio','wall','pantry','toilet','chair','candlepot','holder','hanger', 'picture', 'lantern','tapes','gloves', 'beaky', 'housing','hang', 'box', 'lamp','basket','key','wicker','board','blackboard','fan','drawer','bucket','thermometer','mirror','sponge', 'ladder','sign', 'doorsign','storage','photo', 'bin','hldr', 'doormat','chalk','cushion', 'letter','quilt', 'sketchbook ', 'finish','heart', 'word','ornament','magnets','alarm', 'frame', 'magnet', 'light', 'candle','parasol','towel','paint', 'postage','doorstop','clock','cases','rack','candleholder','cutters']
children_category = ['butterfly','crawlie','carousel','farm','chicken','cactus','dollcraft','monster','carriage','snake','dolphin','fun','gnome','pear','w/chime', 'bird','space','bunny','spinning', 'lolly','dog','ludo','helicopter','cross','spin','tattoo','elephant','teddy','horse','hen','sheep','bingo','figure','stencil','drawing','bunnies','rabbit','bow','cat','ball','alphabet','sandcastle','tatoo','piggy','child','knick','tatoos', 'children', 'dinosaur', 'creatures', 'playhouse', 'fairy','toy', 'school', 'dolls', 'skipping','knick knack', 'dominoes', 'dollhouse','girly','girl','doll','globe','sticker', 'game', 'puzzles','jigsaw','dolly', 'feltcraft','books','toadstools','study','flying','soldier','owl'] 
stationery_category = ['c/cover','punch','notelets','journal','book','sketchbook', 'collage','pencil','sharpener','crayon','tall','stamp','stationery','pencil','writing','envelope','erasers','notepad','colouring','colour','postcard','scetchbook','pen','scissor','ruler','organiser','notebook','note','paper', 'crayons','pencils']
party_category = ['cube','minicards','chest','daisy','flag','disc/mirror','confetti','buddha', 'cocktail', 'hawaiian','star','crawlies','sombrero','bell','straw','badge','pennant','easter','straw','ribbon','party','card','wreath','bunting','pirate', 'santa','garland', 'disco', 'christmas','wrap','gift','baloon', 'egg','chocolate', 'balloon','decoration']
kitchen_category = ['goblet','fruitbowl','jampot','platter','cooking','teatime','toastrack', 'spoon', 'folkart','cup','sugar','squeezer','chopstick','coffee','bowl','teapot','apron','cakestand','crackers','oven','porcelain','herb','bottle','folk','food','biscuit','baking', 'bakelike','fruit','potting','cake','plate', 'beaker','sugar','placemats','match','dish','toast','ducks', 'jug', 'scales','dispencer', 'cup', 'moulds', 'mould', 'saucer', 'plate', 'jam','tea','napkin','pan','bottle','kitchen', 'coaster', 'cutlery','cookie', 'napkins','mug','glass','pot','beaker','lunch','bowl','tablecloth','teaspoon', 'teacup','tissue', 'tray']
beauty_category = ['charm','hairslide','chain','sunglass','braclet','hairclip','sock','necklace+bracelet','jewel','hair','hairclips','clip','lariat','necklace','shirt','shoe', 'bangle','pad','handbag','patch','cream','band','ring','dress','scarf','bracelet','coat','bag','shopper', 'backpack','purse','earring','flannel', 'crystal']
makeup_category = ['lip']
sewing_category = ['sew', 'sewing', 'knitting','bead', 'tinsel']
tech_category = ['charge','mobile','boombox','phone']
outdoor_category = ['sand','aid','vippassport','rubber','forest','rucksack','passport','luggage','warmer', 'plaster','bicycle','umbrella','rain','picnic']


# In[63]:


def lemmatization_func(line):
    lemmatized=lemmatize_text(line)
    if any(word in lemmatized for word in houseware_category):
        return 'houseware'
    elif any(word in lemmatized for word in children_category):
        return 'children'
    elif any(word in lemmatized for word in party_category):
        return 'party'
    elif any(word in lemmatized for word in kitchen_category):
        return 'kitchen'
    elif any(word in lemmatized for word in outdoor_category):
        return 'outdoor'
    elif any(word in lemmatized for word in beauty_category):
        return 'beauty'
    elif any(word in lemmatized for word in stationery_category):
        return 'stationery'
    elif any(word in lemmatized for word in sewing_category):
        return 'sewing'
    elif any(word in lemmatized for word in makeup_category):
        return 'makeup'
    elif any(word in lemmatized for word in tech_category):
        return 'tech'
    else:
        return 'other'


# In[64]:


dataset['category']=dataset['description'].apply(lemmatization_func)


# In[65]:


dataset[dataset['category']=='other']


# In[66]:


dataset['category'].value_counts()


# After all we have 10 categories of items sold in our shop. The range is really wide, we have a lot of small items from  products for house, for cooking needs, for making parties to stationery, hobbies and even technical and makeup items. The biggest category is products used for housing needs 'houseware', the smallest is the 'tech' category. We should study the data by categories more closely by now.

# In[67]:


dataset_new = dataset.drop_duplicates(subset='description', keep="last")


# In[68]:


df_new = df_positive.merge(dataset_new, how='left', on='description')
df_new.head(2)


# In[69]:


df_new.boxplot(by = 'category', column = ['total_inv'], grid=True, figsize=(10,8), showfliers = False)
plt.ylim(0,80)
plt.show();


# <div class="alert alert-warning"> <b>Team Lead comment:</b> Let's see boxplot without outliers :)
# </div>

# Here we see how the categories devided in terms of total_invoices. The most number of outliers has the houseware, children and party group/ The smallest groups ae tech and makeup categories. Also we see that all the categories are grouped within the 20USD total invoice.

# In[71]:


dates_categories = pd.pivot_table(df_new, values='total_inv', index=['month'], columns = ['category'], aggfunc='sum',fill_value=0)
dates_categories


# In[72]:


plt.figure(figsize=(15,8))
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
g = sns.lineplot(data=dates_categories,linewidth=3,markers=True, dashes=False)
plt.ylim(0,550000)
g.ylabels = ['{:,.2f}'.format(y) + 'K' for y in g.get_yticks()/1000]
g.set_yticklabels(g.ylabels)
plt.xticks(rotation = 45)
plt.xlabel("MONTHS")
plt.ylabel("SALES SUM")
plt.title("Total sales per month by product categories", fontsize = 14);


# <div class="alert alert-warning"> <b>Team Lead comment:</b> Graph is hard to read :( Can we work with axis please?
# </div>

# The graph describes us the situation with the sales of our products by months. We can tell that almost every category represents the same trend of grows and falls, but the houseware, stationary, children and party categories show us the better grow, especially we see that the houseware and party categories have a rapid boom of sales starting from September 2019. The others remain steady within the year, the tech category is falling and brings a very few income for the company.

# In[73]:


dynamics_month=((dates_categories - dates_categories.shift(+1))/1000).T
dynamics_month


# In[74]:


plt.figure(figsize=(16,10))
ax = sns.heatmap(dynamics_month, annot=True, square=True, cbar=True, fmt='.2f', cmap='PuOr', xticklabels=True, yticklabels=True, linewidths=.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12, rotation_mode='anchor', ha='right')
plt.title("Total sales dynamics per month by product categories",fontsize=14);


# <div class="alert alert-warning"> <b>Team Lead comment:</b> Let's add labels to the graph
# </div>

# The positive dynamics of total sales is shown to us on this heatmap.  The houseware category of goods has the leading positions here. But also we see that no category has negative dinamics, even when the growth is not that much.

# In[75]:


categories_grouped = (
    df_new.groupby(["category"])
        .agg({"invoiceno": "nunique", "total_inv": "sum", "quantity":"sum"})
        .sort_values(by="category", ascending=False)
        .reset_index()
)
categories_grouped = categories_grouped.head(10).sort_values(by='total_inv',ascending=False)
categories_grouped


# In[76]:


fig = px.bar(categories_grouped, x=categories_grouped.invoiceno, y=categories_grouped.category, title='Number of invoices by the categories',
            color=categories_grouped.category,)
fig.update_xaxes(tickangle=45)
fig.show() 


# As fow number of invoices the houseware is also the leader, the 2 and 3 rd ones are goods for kitchen and party. That means that customers order really a lot of products within these categories and again the tech category is the smallest. In my opininon that's because people prefer to buy this type of products in the specialised shops and the range of goods of this field is really small and unneccesary for us. 

# In[77]:


fig3=px.pie(categories_grouped, values=categories_grouped.total_inv, names=categories_grouped.category,
           color=categories_grouped.category,
        color_discrete_sequence=px.colors.qualitative.G10)
fig3.update_layout(title ="Proportions of the product categories by total invoice" )
fig3.show();


# The piechart provides us with the information about proportions that diffenet categories have. As we see 60,0% relies to houseware, 13,5 % to kitchen categories and 9.99% to party. However less then 1% has the outdoor, sewing, makeup and tech categories.

# In[78]:


price_df_1 =  pd.pivot_table(df_new, values='total_inv', index=['category'], columns = ['price_segment'], aggfunc='sum',fill_value=0)
price_df_1


# All the goods refer to the economy segment, the exseption is houseware that presents also both in medium high and medium low categories. 

# In[79]:


categories_grouped1 = (
    df_new.groupby(["category"])
        .agg({"unitprice": "mean", "total_inv": "mean"})
        .sort_values(by="category", ascending=False)
        .reset_index()
)
categories_grouped1 = categories_grouped1.head(10).sort_values(by='total_inv',ascending=False)
categories_grouped1


# In[80]:


import plotly.express as px

fig = px.scatter(categories_grouped1, x="category", y="total_inv",size='unitprice',
                 color="total_inv", color_continuous_scale=px.colors.sequential.Viridis,title="Average unitprice and average total invoice within categories")

fig.show()


# Here we see the opening of the analysis that the unit price in the makeup category is small, but the average total invoice in this group is the largest one. We can predict that the customer can be interested if the range of the products in this category can be expanded and that can bring us a new flow of income. Also as we see below, the minimum unitprice in the makeup category is the largest, the houseware category presents here the cheapest item within all categories.

# In[81]:


sample1=df_new.groupby(['category'])['unitprice'].min()
sample1.sort_values()


# In[82]:


sample2=df_new.groupby(['category'])['unitprice'].mean()
sample2.sort_values()


# <div class="alert alert-success"> <b>Team Lead comment:</b> Nice graphs!
# </div>

# ## Testing Hypothesis

# ### Now let's move to the last step and test the hypotheses:
#  
# 	The average revenue from houseware category and the makeup categories differs.
#     
#     We consider that the null hypothesis is effectively stating that a difference between comparing situation is equal to zero, that the 2 parameteres that we are comparing are equal.

# #### We are going to check our hypothese
# **H0 = the average revenue from houseware category and the makeup categories is the same.\
# H1 = the average revenue from houseware category and the makeup categories differs**

# In[83]:


makeup_cat = df_new[df_new['category'].str.contains('makeup')]


# In[84]:


makeup_sample = makeup_cat['total_inv']


# In[85]:


houseware_cat = df_new[df_new['category'].str.contains('houseware')]


# In[86]:


houseware_sample = houseware_cat['total_inv']


# In[87]:


alpha = .05 

results = stats.shapiro(makeup_sample)
p_value = results[1] 

print('p-value: ', p_value)

if (p_value < alpha):
    print("Null hypothesis rejected: the distribution is not normal")
else:
    print("Failed to reject the null hypothesis: the distribution seems to be normal")


# In[88]:


alpha = .05 

results = stats.shapiro(houseware_sample)
p_value = results[1] 

print('p-value: ', p_value)

if (p_value < alpha):
    print("Null hypothesis rejected: the distribution is not normal")
else:
    print("Failed to reject the null hypothesis: the distribution seems to be normal") 


# As the distribution is not normal, we can use the Mann Whitneyu test.Let's found out if there is a signicance difference in the conversion rates between 2 groups
# Hypothesis
# H(0):  There is not a significant difference between the metrics
# H(1): There is no reason to consider the metrics are the same

# In[89]:


alpha=0.05
p_value = stats.mannwhitneyu(makeup_sample, houseware_sample)[1]
if (p_value < alpha):
    print("Rejecting the null hypothesis: there is not a significant difference between the proportions")
else:
    print("Failed to reject the null hypothesis: there is no reason to consider the proportions different") 


# In[90]:


print("{0:.5f}".format(stats.mannwhitneyu(makeup_sample, houseware_sample)[1]))
print("{0:.3f}".format(makeup_sample.mean()/houseware_sample.mean()-1))


# The p_value is lower than 0.05, the data analysis points that the groups average revenue has statistically significant differences. The relative difference between 2 groups is 9.4%.
# We can conclude that there is difference between 2 groups in terms of revenue and the distribution inside two groups is not normal.

# <div class="alert alert-warning"> <b>Team Lead comment:</b> Please check the normality of your distributions before choosing the test
# </div>

# # <center>Conclusions<center>

# The analysis of the data in the e-commerce firm contained an information about all the invoices, products, quantity and unitprice for allmost a year starting from 12.2018 and ending in 11.2019. The data was cleaned of the missing values ( description and customer id), duplicated rows and few unnecessary dates ( 1 day in october, 2018 and a week in the december, 2019)\
# We figured out that the most goods that we sell refer to the economy segment, we don't sell exclusive or expensive items. We position ourselves as the shop where we can find a really huge range of products (3900 unique items) for a little price starting from 0.001$. The average total invoice that we have is equall to 19 USD. The average number of monthly purchases is 1584.\
# The total dynamics of revenue growths and grows of purchases is positive, but we had some losses in February and April 2019. It is an issue we should check with the command to understand the reason of such fall.\
# After using lemmatization function we have 10 categories of goods that we sell. 
# The investigation of this categories showed us that the most popular, the most profitable and frequently ordered one is the houseware category, that provides a lot of stuff for the house and the garden, the second top category is the kitchen category that consists of items for cooking and serving, the goods for children are also very important and widely presented in our shop.\
# However we see, that some categories like goods for sewing, the technical items and outdor categories take less then 1% in the total invoice proportions. We can assume that people don't really often buy these goods, we don't have enough items to sell in technical category, the range is really narrow there. Seems like customers prefer to buy technical items in specialised stores.\
# So in my opinion we should think off removing these categories from our product range, as they are sold really bad and as we are the shop that provides e-commerse, sometimes it is really hard to scroll through such a big amount of items. However we see, that the average invoice in makeup category provides us a very good result, the unitprice in this category is the largest one and the average total revenue shows us the good result, we can start expanding the choice in this category if the client has the interest in it because as for now we only have 1 item presented in our online store.

# https://cxl.com/blog/product-pricing-strategies-and-techniques/ \
# https://www.sigmacomputing.com/blog/how-data-drives-a-successful-product-mix-analysis/ \
# https://www.machinelearningplus.com/nlp/lemmatization-examples-python \
# https://www.sisense.com/blog/10-useful-ways-visualize-data-examples/#Bubble \
# https://www.toptal.com/product-managers/data/product-hypothesis-testing \
# https://medium.com/swlh/product-sales-analysis-using-python-863b29026957 \
# https://www.dotactiv.com/blog/product-range-review"
