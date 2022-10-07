#!/usr/bin/env python
# coding: utf-8

# # PyTorch Basics: Tensors & Gradients
# 
# ### Part 1 of "Deep Learning with Pytorch: Zero to GANs"
# 
# This tutorial series is a hands-on beginner-friendly introduction to deep learning using [PyTorch](https://pytorch.org), an open-source neural networks library. These tutorials take a practical and coding-focused approach. The best way to learn the material is to execute the code and experiment with it yourself. Check out the full series here:
# 
# 1. [PyTorch Basics: Tensors & Gradients](https://jovian.ai/aakashns/01-pytorch-basics)
# 2. [Gradient Descent & Linear Regression](https://jovian.ai/aakashns/02-linear-regression)
# 3. [Working with Images & Logistic Regression](https://jovian.ai/aakashns/03-logistic-regression) 
# 4. [Training Deep Neural Networks on a GPU](https://jovian.ai/aakashns/04-feedforward-nn)
# 5. [Image Classification using Convolutional Neural Networks](https://jovian.ai/aakashns/05-cifar10-cnn)
# 6. [Data Augmentation, Regularization and ResNets](https://jovian.ai/aakashns/05b-cifar10-resnet)
# 7. [Generating Images using Generative Adversarial Networks](https://jovian.ai/aakashns/06b-anime-dcgan/)

# This tutorial covers the following topics:
# 
# * Introductions to PyTorch tensors
# * Tensor operations and gradients
# * Interoperability between PyTorch and Numpy
# * How to use the PyTorch documentation site

# ### Prerequisites
# 
# If you're just getting started with data science and deep learning, then this tutorial series is for you. You just need to know the following:
# 
# - Basic Programming with Python ([variables](https://jovian.ai/aakashns/first-steps-with-python), [data types](https://jovian.ai/aakashns/python-variables-and-data-types), [loops](https://jovian.ai/aakashns/python-branching-and-loops), [functions](https://jovian.ai/aakashns/python-functions-and-scope) etc.)
# - Some high school mathematics ([vectors, matrices](https://www.youtube.com/watch?v=0oGJTQCy4cQ&list=PLSQl0a2vh4HCs4zPpOEdF2GuydqS90Yb6), [derivatives](https://www.youtube.com/watch?v=N2PpRnFqnqY) and [probability](https://www.youtube.com/watch?v=uzkc-qNVoOk))
# - No prior knowledge of data science or deep learning is required
# 
# We'll cover any additional mathematical and theoretical concepts we need as we go along.
# 
# 

# ### How to run the code
# 
# This tutorial is an executable [Jupyter notebook](https://jupyter.org) hosted on [Jovian](https://www.jovian.ai) (don't worry if these terms seem unfamiliar; we'll learn more about them soon). You can _run_ this tutorial and experiment with the code examples in a couple of ways: *using free online resources* (recommended) or *on your computer*.
# 
# #### Option 1: Running using free online resources (1-click, recommended)
# 
# The easiest way to start executing the code is to click the **Run** button at the top of this page and select **Run on Colab**. [Google Colab](https://colab.research.google.com) is a free online platform for running Jupyter notebooks using Google's cloud infrastructure. You can also select "Run on Binder" or "Run on Kaggle" if you face issues running the notebook on Google Colab.
# 
# 
# #### Option 2: Running on your computer locally
# 
# To run the code on your computer locally, you'll need to set up [Python](https://www.python.org), download the notebook and install the required libraries. We recommend using the [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) distribution of Python. Click the **Run** button at the top of this page, select the **Run Locally** option, and follow the instructions.
# 
# >  **Jupyter Notebooks**: This tutorial is a [Jupyter notebook](https://jupyter.org) - a document made of _cells_. Each cell can contain code written in Python or explanations in plain English. You can execute code cells and view the results, e.g., numbers, messages, graphs, tables, files, etc. instantly within the notebook. Jupyter is a powerful platform for experimentation and analysis. Don't be afraid to mess around with the code & break things - you'll learn a lot by encountering and fixing errors. You can use the "Kernel > Restart & Clear Output" or "Edit > Clear Outputs" menu option to clear all outputs and start again from the top.

# Before we begin, we need to install the required libraries. The installation of PyTorch may differ based on your operating system / cloud environment. You can find detailed installation instructions here: https://pytorch.org .

# In[34]:


# Uncomment and run the appropriate command for your operating system, if required

# Linux / Binder
# !pip install numpy torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Windows
# !pip install numpy torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# MacOS
# !pip install numpy torch torchvision torchaudio


# Let's import the `torch` module to get started.

# In[1]:


import torch


# ## Tensors
# 
# At its core, PyTorch is a library for processing tensors. A tensor is a number, vector, matrix, or any n-dimensional array. Let's create a tensor with a single number.

# In[2]:


# Number
t1 = torch.tensor(4.)
t1


# `4.` is a shorthand for `4.0`. It is used to indicate to Python (and PyTorch) that you want to create a floating-point number. We can verify this by checking the `dtype` attribute of our tensor.

# In[4]:


t1.dtype


# Let's try creating more complex tensors.

# In[3]:


# Vector
t2 = torch.tensor([1., 2, 3, 4])
t2


# In[4]:


# Matrix
t3 = torch.tensor([[5., 6], 
                   [7, 8], 
                   [9, 10]])
t3


# In[5]:


# 3-dimensional array
t4 = torch.tensor([
    [[11, 12, 13], 
     [13, 14, 15]], 
    [[15, 16, 17], 
     [17, 18, 19.]]])
t4


# Tensors can have any number of dimensions and different lengths along each dimension. We can inspect the length along each dimension using the `.shape` property of a tensor.

# In[6]:


print(t1)
t1.shape


# In[7]:


print(t2)
t2.shape


# In[8]:


print(t3)
t3.shape


# In[9]:


print(t4)
t4.shape


# Note that it's not possible to create tensors with an improper shape.

# In[10]:


# Matrix
t5 = torch.tensor([[5., 6, 11], 
                   [7, 8], 
                   [9, 10]])
t5


# A `ValueError` is thrown because the lengths of the rows `[5., 6, 11]` and `[7, 8]` don't match.

# ## Tensor operations and gradients
# 
# We can combine tensors with the usual arithmetic operations. Let's look at an example:

# In[11]:


# Create tensors.
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
x, w, b


# We've created three tensors: `x`, `w`, and `b`, all numbers. `w` and `b` have an additional parameter `requires_grad` set to `True`. We'll see what it does in just a moment. 
# 
# Let's create a new tensor `y` by combining these tensors.

# In[12]:


# Arithmetic operations
y = w * x + b
y


# As expected, `y` is a tensor with the value `3 * 4 + 5 = 17`. What makes PyTorch unique is that we can automatically compute the derivative of `y` w.r.t. the tensors that have `requires_grad` set to `True` i.e. w and b. This feature of PyTorch is called _autograd_ (automatic gradients).
# 
# To compute the derivatives, we can invoke the `.backward` method on our result `y`.

# In[13]:


# Compute derivatives
y.backward()


# The derivatives of `y` with respect to the input tensors are stored in the `.grad` property of the respective tensors.

# In[14]:


# Display gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)


# As expected, `dy/dw` has the same value as `x`, i.e., `3`, and `dy/db` has the value `1`. Note that `x.grad` is `None` because `x` doesn't have `requires_grad` set to `True`. 
# 
# The "grad" in `w.grad` is short for _gradient_, which is another term for derivative. The term _gradient_ is primarily used while dealing with vectors and matrices.

# ## Tensor functions
# 
# Apart from arithmetic operations, the `torch` module also contains many functions for creating and manipulating tensors. Let's look at some examples.

# In[17]:


# Create a tensor with a fixed value for every element
t6 = torch.full((3, 2), 42)
t6


# In[18]:


# Concatenate two tensors with compatible shapes
t7 = torch.cat((t3, t6))
t7


# In[19]:


# Compute the sin of each element
t8 = torch.sin(t7)
t8


# In[20]:


# Change the shape of a tensor
t9 = t8.reshape(3, 2, 2)
t9


# You can learn more about tensor operations here: https://pytorch.org/docs/stable/torch.html . Experiment with some more tensor functions and operations using the empty cells below.

# In[22]:


t10=torch.tanh(t7)
t10


# In[23]:


t11=torch.cos(t6)
t11


# In[24]:


t11.reshape(2,3)


# In[30]:


t12=torch.cat((t10,t11))
t12


# In[ ]:





# In[ ]:





# ## Interoperability with Numpy
# 
# [Numpy](http://www.numpy.org/) is a popular open-source library used for mathematical and scientific computing in Python. It enables efficient operations on large multi-dimensional arrays and has a vast ecosystem of supporting libraries, including:
# 
# * [Pandas](https://pandas.pydata.org/) for file I/O and data analysis
# * [Matplotlib](https://matplotlib.org/) for plotting and visualization
# * [OpenCV](https://opencv.org/) for image and video processing
# 
# 
# If you're interested in learning more about Numpy and other data science libraries in Python, check out this tutorial series: https://jovian.ai/aakashns/python-numerical-computing-with-numpy .
# 
# Instead of reinventing the wheel, PyTorch interoperates well with Numpy to leverage its existing ecosystem of tools and libraries.

# Here's how we create an array in Numpy:

# In[31]:


import numpy as np

x = np.array([[1, 2], [3, 4.]])
x


# We can convert a Numpy array to a PyTorch tensor using `torch.from_numpy`.

# In[32]:


# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)
y


# Let's verify that the numpy array and torch tensor have similar data types.

# In[33]:


x.dtype, y.dtype


# We can convert a PyTorch tensor to a Numpy array using the `.numpy` method of a tensor.

# In[34]:


# Convert a torch tensor to a numpy array
z = y.numpy()
z


# The interoperability between PyTorch and Numpy is essential because most datasets you'll work with will likely be read and preprocessed as Numpy arrays.
# 
# You might wonder why we need a library like PyTorch at all since Numpy already provides data structures and utilities for working with multi-dimensional numeric data. There are two main reasons:
# 
# 1. **Autograd**: The ability to automatically compute gradients for tensor operations is essential for training deep learning models.
# 2. **GPU support**: While working with massive datasets and large models, PyTorch tensor operations can be performed efficiently using a Graphics Processing Unit (GPU). Computations that might typically take hours can be completed within minutes using GPUs.
# 
# We'll leverage both these features of PyTorch extensively in this tutorial series.

# ## Save and upload your notebook
# 
# Whether you're running this Jupyter notebook online or on your computer, it's essential to save your work from time to time. You can continue working on a saved notebook later or share it with friends and colleagues to let them execute your code. [Jovian](https://jovian.ai/platform-features) offers an easy way of saving and sharing your Jupyter notebooks online.
# 
# First, you need to install the Jovian python library if it isn't already installed.

# In[25]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[26]:


import jovian


# In[33]:


jovian.commit(project='01-pytorch-basics')


# > The first time you run `jovian.commit`, you may be asked to provide an _API Key_ to securely upload the notebook to your Jovian account. You can get the API key from your [Jovian profile page](https://jovian.ai) after logging in / signing up.
# 
# `jovian.commit` uploads the notebook to your Jovian account, captures the Python environment, and creates a shareable link for your notebook, as shown above. You can use this link to share your work and let anyone (including you) run your notebooks and reproduce your work. Jovian also includes a powerful commenting interface, so you can discuss & comment on specific parts of your notebook:
# 
# ![https://jovian.ai/docs/user-guide/upload.html](https://i.imgur.com/kxx3pqM.png)
# 
# You can do a lot more with the `jovian` Python library. Visit the documentation site to learn more: https://jovian.ai/docs/index.html

# ## Summary and Further Reading
# 
# Try out this assignment to learn more about tensor operations in PyTorch: https://jovian.ai/aakashns/01-tensor-operations
# 
# 
# This tutorial covers the following topics:
# 
# * Introductions to PyTorch tensors
# * Tensor operations and gradients
# * Interoperability between PyTorch and Numpy
# 
# 
# You can learn more about PyTorch tensors here: https://pytorch.org/docs/stable/tensors.html. 
# 
# 
# The material in this series is inspired by:
# 
# * [PyTorch Tutorial for Deep Learning Researchers](https://github.com/yunjey/pytorch-tutorial) by Yunjey Choi 
# * [FastAI development notebooks](https://github.com/fastai/fastai_docs/tree/master/dev_nb) by Jeremy Howard. 
# 
# With this, we complete our discussion of tensors and gradients in PyTorch, and we're ready to move on to the next topic: [Gradient Descent & Linear Regression](https://jovian.ai/aakashns/02-linear-regression).

# ## Questions for Review
# 
# Try answering the following questions to test your understanding of the topics covered in this notebook:
# 
# 1. What is PyTorch?
# 2. What is a Jupyter notebook?
# 3. What is Google Colab?
# 4. How do you install PyTorch?
# 5. How do you import the `torch` module?
# 6. What is a vector? Give an example.
# 7. What is a matrix? Give an example.
# 8. What is a tensor?
# 9. How do you create a PyTorch tensor? Illustrate with examples.
# 10. What is the difference between a tensor and a vector or a matrix?
# 11. Is every tensor a matrix?
# 12. Is every matrix a tensor?
# 13. What does the `dtype` property of a tensor represent?
# 14. Is it possible to create a tensor with elements of different data types?
# 15. How do you inspect the number of dimensions of a tensor and the length along each dimension?
# 16. Is it possible to create a tensor with the values `[[1, 2, 3], [4, 5]]`? Why or why not?
# 17. How do you perform arithmetic operations on tensors? Illustrate with examples?
# 18. What happens if you specify `requires_grad=True` while creating a tensor? Illustrate with an example.
# 19. What is autograd in PyTorch? How is it useful?
# 20. What happens when you invoke  the `backward` method of a tensor?
# 21. How do you check the derivates of a result tensor w.r.t. the tensors used to compute its value?
# 22. Give some examples of functions available in the `torch` module for creating tensors.
# 23. Give some examples of functions available in the `torch` module for performing mathematical operations on tensors.
# 24. Where can you find the list of tensor operations available in PyTorch?
# 25. What is Numpy?
# 26. How do you create a Numpy array?
# 27. How do you create a PyTorch tensor using a Numpy array?
# 28. How do you create a Numpy array using a PyTorch tensor?
# 29. Why is interoperability between PyTorch and Numpy important?
# 30. What is the purpose of a library like PyTorch if Numpy already provides data structures and utilities to with multi-dimensional numeric data?
# 31. What is Jovian?
# 32. How do you upload your notebooks to Jovian using `jovian.commit` ?
# 

# In[ ]:




