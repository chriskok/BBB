from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys

options = Options()
options.add_argument("start-maximized")
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
# driver.get("https://www.google.com")

# driver.get("https://www.gradescope.com/courses/474923/assignments/2501989/submissions/154243492")
driver.get("https://www.gradescope.com/auth/saml/umich")
print(driver.title)
# search_bar = driver.find_element_by_name("q")
# search_bar.clear()
# search_bar.send_keys("getting started with python")
# search_bar.send_keys(Keys.RETURN)
# print(driver.current_url)
# driver.close()

# find username/email field and send the username itself to the input field
driver.find_element("id", "login").send_keys('chriskok')
# find password input field and insert password as well
driver.find_element("id", "password").send_keys('AppleSeahorse_01')
# click login button
driver.find_element("id", "loginSubmit").click()