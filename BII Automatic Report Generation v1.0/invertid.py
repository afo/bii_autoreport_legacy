

def invertid(user_id):
	code = ""
	charachters = ".@_-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!#$%&'^`"
        key = "aAbBcCdDeEfFgG0123456789hHiIjJkKlLmMnNoOp@PqQrRsStTuUvVw._-WxXyYzZ!#$%&'^`"

	i=0
	for c in user_id:
		n = key.find(c)
		code += charachters[n]

	return code
