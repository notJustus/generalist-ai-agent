# Evolutionary Computing

## Setting up Git

1. **Clone** the repo locally: `git clone https://github.com/notJustus/ml-pneumonia-classification`

2. Create a feature branch and checkout to that branch: `git checkout -b <feature-[name]>`

3. Create a folder in the root of the working directory called `/data` 

4. Add all the xray data files in this folder

Run jupyter notebook with: `jupyter notebook`

## After working on the project

After you're done with your work just **commit & push** the changes:

`git add .`

`git commit -m "Implemented ..."`

`git push origin <your-branch-nameg>`


> If you just experimented a bit just leave it like this.

> If you made some progress create a new pull request


### When you work on the project again make sure to always **pull** the latest version before starting to code again

`git checkout main`

`git pull origin main`

`git checkout dev`

`git pull origin dev`

After that switch to your branch again and work in it just like before:

`git checkout <yourBranch>`

## Merging with dev

When you implement something that is complete, you can make a **pull request** to merge it with *dev*

1. Go on Github and switch to your feature branch
2. Click on the yellow banner that says **"Compare & pull request"**
3. For the base select *dev*
4. For compare select your own feature branch
5. (Optional) You can add some description
6. When done, press on **"Create pull request"**


## Some Useful Commands

`git status` 
> To check what files still need to be added and committed

`git branch`
> To check which branch you are currently on
