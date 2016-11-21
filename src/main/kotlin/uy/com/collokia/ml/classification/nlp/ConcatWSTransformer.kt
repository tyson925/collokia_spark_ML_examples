package uy.com.collokia.ml.classification.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructType
import java.util.*

class ConcatWSTransformer : Transformer {

    var inputColNames: Array<String>
    var outputColName : String

    constructor() {
        inputColNames = arrayOf("content_1","content_2")
        outputColName = "content_ws"
    }


    fun getInputCols(): Array<String> {
        return inputColNames
    }

    fun setInputCols(inputCols: Array<String>): ConcatWSTransformer {
        this.inputColNames = inputCols
        return this
    }


    fun getOutputCol(): String {
        return outputColName
    }

    fun setOutputCol(outputCol : String): ConcatWSTransformer {
        this.outputColName = outputCol
        return this
    }


    override fun uid(): String {
        return UUID.randomUUID().toString()
    }

    override fun copy(p0: ParamMap?): Transformer {
        return ConcatWSTransformer()
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row>? {
        return dataset?.let {
            val outputSchema = transformSchema(dataset.schema())
            val metadata = outputSchema.apply(outputColName).metadata()

            dataset.select(dataset.col("*"),
                    functions.split(functions.concat_ws(" ",dataset.col(inputColNames[0]),dataset.col(inputColNames[1]))," ").`as`(outputColName))
        }

    }

    override fun transformSchema(schema: StructType?): StructType {

        val inputType = schema?.apply(schema.fieldIndex(inputColNames[0]))

        val inputTypeMetaData = inputType?.dataType()
        val refType = DataTypes.createArrayType(DataTypes.StringType)

        if (inputTypeMetaData is ArrayType){
            println("Input type must be ArrayType(StringType) but got $inputTypeMetaData.")
        }
        return SchemaUtils.appendColumn(schema, outputColName,inputType?.dataType(),inputType?.nullable() ?: false)
    }

}
