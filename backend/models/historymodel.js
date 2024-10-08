const { DataTypes } = require('sequelize');
const sequelize = require('../config/db');

const History = sequelize.define('History', {
    userId: {
        type: DataTypes.INTEGER,
        allowNull: false,
    },
    featureVectors: {
        type: DataTypes.TEXT,  
        allowNull: false,
    },
    classification: {
        type: DataTypes.STRING,
        allowNull: false,
    },
    createdAt: {
        type: DataTypes.DATE,
        defaultValue: DataTypes.NOW,
    },
});

module.exports = History;