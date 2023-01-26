#ensuring the local repository is up to date with the one on github
git status
#saving the current date and time into a file called "version" inside the repository
date > /data/class/cogs106/kvarieur/cogs106repo/version
#commiting the changes to the new file
git add .
git commit -m "new version"
#updating github repository with new file
git push -u origin main
