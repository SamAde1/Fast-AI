#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision import *


# In[2]:


classes = ['Gorilla','Gibbon','Chimpanzee','Orangutan']


# In[4]:


for c in classes:
    print(c)
    folder = c
    file=c+".txt"
    path = Path('Data')
    dest = path/folder
    #dest.mkdir(parents=True, exist_ok=True)
    #download_images(file, dest, max_pics=100, max_workers=0)


# In[5]:


for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)


# In[18]:


np.random.seed(40)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, bs=24,
        ds_tfms=get_transforms(), size=224, num_workers=4,no_check=True).normalize(imagenet_stats)


# In[19]:


data.classes


# In[20]:


data.show_batch(rows=3, figsize=(7,8))


# In[21]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[22]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[23]:


learn.fit_one_cycle(4)


# In[24]:


learn.save('stage-1')


# In[25]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[26]:


interp.plot_top_losses(9, figsize=(15,11))


# In[27]:


learn.unfreeze()


# In[30]:


learn.lr_find()


# In[31]:


learn.recorder.plot()


# In[32]:


learn.fit_one_cycle(2, max_lr=slice(1e-7,3e-5))


# In[33]:


learn.save('stage-2')


# In[34]:


learn.load('stage-2');


# In[35]:


interp = ClassificationInterpretation.from_learner(learn)


# In[36]:


interp.plot_confusion_matrix()


# In[37]:


from fastai.widgets import *


# In[38]:


db = (ImageList.from_folder(path)
                   .split_none()
                   .label_from_folder()
                   .transform(get_transforms(), size=224)
                   .databunch()
     )


# In[39]:


learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)

learn_cln.load('stage-2');


# In[40]:


ds, idxs = DatasetFormatter().from_toplosses(learn_cln)


# In[41]:


ImageCleaner(ds, idxs, path)


# In[42]:


ds, idxs = DatasetFormatter().from_similars(learn_cln)


# In[43]:


ImageCleaner(ds, idxs, path, duplicates=True)


# In[44]:


[ ]:
learn.export()


# In[45]:



learn.export()


# In[46]:


defaults.device = torch.device('cpu')


# In[50]:


img = open_image(path/'Gorilla'/'00000036.jpg')
img


# In[51]:


learn = load_learner(path)


# In[52]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# In[ ]:




