from api.init import app
from api.basicRoutes import *
from api.apiRoutes import *
from api.errorHandelers import *


if __name__ == '__main__':
    app.run(debug=True)
