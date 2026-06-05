interface Feature {
  feature: string
  importance: number
}

interface Props {
  features: Feature[]
}

export default function FeatureImportance({ features }: Props) {
  const maxImportance = Math.max(...features.map(f => f.importance))

  return (
    <div className="card">
      <h2 className="text-2xl font-semibold mb-6">Feature Importance</h2>
      <div className="space-y-4">
        {features.map((feature, i) => (
          <div key={i}>
            <div className="flex justify-between mb-1">
              <span className="font-medium">{feature.feature}</span>
              <span className="text-sm text-gray-600">
                {(feature.importance * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-gray-200 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-blue-400 to-blue-600 h-3 rounded-full"
                style={{
                  width: `${(feature.importance / maxImportance) * 100}%`,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}