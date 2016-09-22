package uy.com.collokia.ml.classification.nlp

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.DataTypes
import scala.Function1
import scala.collection.mutable.WrappedArray

class OwnNGram : UnaryTransformer<WrappedArray<String>, Array<String>, OwnNGram>() {

    override fun createTransformFunc(): Function1<WrappedArray<String>, Array<String>> {
        return ConvertFunction()
    }


    override fun outputDataType(): DataType {
        return DataTypes.createArrayType(DataTypes.StringType)
    }

    override fun uid(): String {
        //return UUID.randomUUID().toString()
        return "uid1111111"
    }

}

