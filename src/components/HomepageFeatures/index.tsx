import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<"svg">>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: "高效编程",
    Svg: require("@site/static/img/1-High-efficiency.png").default,
    description: (
      <>
        vLLM 采用独特的 PagedAttention 技术和动态批处理机制，且支持并行采样、波束搜索等多种解码算法，极大提升了服务吞吐量和响应速度
      </>
    ),
  },
  {
    title: "实时编译",
    Svg: require("@site/static/img/2Memory-Management.png").default,
    description: (
      <>创新的内存管理与执行架构，通过将 kv 缓存分割为若干块进行精细管理，把内存浪费控制在序列的最后一块，能实现接近最优的内存使用且支持内存共享，浪费比例低至不到 4%
      </>
    ),
  },
  {
    title: "灵活的迭代空间结构",
    Svg: require("@site/static/img/3-Flexible-user-friendly.png").default,
    description: (
      <>
        vLLM 可无缝集成各类模型，兼容 NVIDIA、AMD、Intel 等多种硬件平台 GPU 和 CPU，并提供简洁的接口和文档，易于上手
      </>
    ),
  },
];

function Feature({ title, Svg, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
