
def reassign_comp(comp):	
	if comp == 'all':
  		comp = 'All'
	elif comp == 'male':
  		comp = 'Male'
	elif comp == 'female':
  		comp = 'Female'
	elif comp == 'study1':
  		comp = "High School"
	elif comp == 'study2':
  		comp = "Undergrad"
	elif comp == 'study3':
  		comp = "Grad or higher ed"
  return comp