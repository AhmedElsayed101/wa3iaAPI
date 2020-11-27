from flask_script import Manager
# from flask_migrate import Migrate, MigrateCommand

from init import app

# from database.models import db

from api.basicRoutes import *
from api.apiRoutes import *
from api.errorHandelers import *

# migrate = Migrate(app, db)
manager = Manager(app)

# manager.add_command('db', MigrateCommand)


if __name__ == '__main__':
    manager.run()